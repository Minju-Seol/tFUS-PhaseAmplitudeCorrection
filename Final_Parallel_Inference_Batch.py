import torch
import time
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from defining_fcns import extract_u_centered_cube, sample_hu_line

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2 ** torch.arange(0, num_frequencies).float()

    def forward(self, x):
        x = x.unsqueeze(-1) * self.freq_bands.to(x.device) * math.pi * 2
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(x.shape[0], -1)

class TransducerMultiHeadModelAmp(nn.Module):
    def __init__(self, num_fourier_freqs=16):
        super().__init__()
        self.ff = FourierFeatures(7, num_frequencies=num_fourier_freqs)
        self.amp_head = nn.Sequential(nn.Linear(197, 256),
                                      nn.GELU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(256, 128),
                                      nn.GELU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(128, 64),
                                      nn.GELU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(64, 1))
        
    def forward(self, target_xyz, trans_xyz, out_inter, in_inter, hu_line, td_idx):
        so_mm = target_xyz*0.001 - torch.tensor([0.132, 0.132, 0.092]).to(device)
        td_mm = trans_xyz *0.001 - torch.tensor([0.132, 0.132, 0.092]).to(device)
        ff_input = torch.cat([td_mm, so_mm],dim=-1)

        out_inter = out_inter.float()
        in_inter = in_inter.float()

        dist_out = torch.norm(out_inter - trans_xyz, dim=-1, keepdim=True)
        dist_out = dist_out * 0.001
        
        dist_thru = torch.norm(out_inter - in_inter, dim=-1, keepdim=True)
        dist_thru = dist_thru * 0.001
        
        dist_in = torch.norm(in_inter - target_xyz, dim=-1, keepdim=True)
        dist_in = dist_in * 0.001
        
        tof = (dist_out / 1500) + (dist_thru / 2262) + (dist_in / 1500)
        tof = tof*1000

        enc_geo = self.ff(ff_input)
        dist_feat = torch.cat([dist_out, dist_thru, dist_in], dim=-1)
        hu_mean = hu_line.mean(dim=1, keepdim=True)
        geo_feat = torch.cat([enc_geo, dist_feat, tof, hu_mean], dim=-1)
        pred_amp = self.amp_head(geo_feat)
        return pred_amp

class PhaseClassifier(nn.Module):
    def __init__(self, dim, num_bins=314, p=0.3):
        super().__init__()
        self.in_ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.out = nn.Linear(256, num_bins)

        k = torch.arange(num_bins).float()
        centers = -math.pi + (2*math.pi)*(k + 0.5)/num_bins
        self.register_buffer('bin_centers', centers)
        self.register_buffer('sin_c', centers.sin())
        self.register_buffer('cos_c', centers.cos())

    def forward(self, x, temperature: float = 1.0):
        h = self.mlp(self.in_ln(x))
        logits = self.out(h)
        probs = F.softmax(logits / temperature, dim=-1)

        s = torch.sum(probs * self.sin_c, dim=-1, keepdim=True)
        c = torch.sum(probs * self.cos_c, dim=-1, keepdim=True)
        phi_hat = torch.atan2(s, c)

        return phi_hat, logits

class TDBias(nn.Module):
    def __init__(self, n_td):
        super().__init__()
        self.bias = nn.Embedding(n_td, 1)
        nn.init.zeros_(self.bias.weight)
    def forward(self, td_id):   
        return self.bias(td_id) 
    
class TransducerMultiHeadModelPhase(nn.Module):
    def __init__(self, num_fourier_freqs=16, num_bins=314):
        super().__init__()
        self.ff = FourierFeatures(6, num_frequencies=num_fourier_freqs) 
        self.phase_head = PhaseClassifier(197, num_bins=num_bins)
        self.td_bias = TDBias(n_td=1024) 

    def forward(self, target_xyz, trans_xyz, out_inter, in_inter, hu_line, td_idx):
        so_mm = target_xyz*0.001 - torch.tensor([0.132, 0.132, 0.092]).to(device)
        td_mm = trans_xyz *0.001 - torch.tensor([0.132, 0.132, 0.092]).to(device)
        
        ff_input = torch.cat([td_mm, so_mm],dim=-1)
        if ff_input.dim() == 3 and ff_input.shape[0] == 1:
            ff_input = ff_input.squeeze(0)
        
        out_inter = out_inter.float()
        in_inter = in_inter.float()
        
        dist_out = torch.norm(out_inter - trans_xyz, dim=-1, keepdim=True)
        dist_out = dist_out * 0.001
        
        dist_thru = torch.norm(out_inter - in_inter, dim=-1, keepdim=True)
        dist_thru = dist_thru * 0.001
        
        dist_in = torch.norm(in_inter - target_xyz, dim=-1, keepdim=True)
        dist_in = dist_in * 0.001
        tof = (dist_out / 1500) + (dist_thru / 2262) + (dist_in / 1500)

        tof = tof*1000

        enc_geo = self.ff(ff_input)
        dist_feat = torch.cat([dist_out, dist_thru, dist_in], dim=-1)
        
        hu_mean = hu_line.mean(dim=1, keepdim=True)
        geo_feat = torch.cat([enc_geo, dist_feat, tof, hu_mean], dim=-1)
        pred_phase, logits = self.phase_head(geo_feat)

        return pred_phase, logits
'''
===============================================================================================================
                                                Data setting
===============================================================================================================
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_skull_idx = 0
aidx = 1

data = torch.load("D:/multielement/successful_codes/github upload/repo_example_data.pt")
skull = data['skull']
td_vxl = data['td_vxl']
so_vxl = data['so_vxl']
ph_list = data['ph_list']
amp_list = data['amp_list']

nbr_td = 1024
num_bins = 314
bin_edges = torch.linspace(-np.pi, np.pi, num_bins + 1)

ph_list = ph_list.to(device)
bin_edges = bin_edges.to(device) 
target_class = torch.bucketize(ph_list, bin_edges, right=True) - 1
target_class = torch.clamp(target_class, 0, num_bins - 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # [360]

all_so = list(range(1000))
extra_ind = list([0,9,90,99,499,900,909,990,999])
remain = [i for i in all_so if i not in extra_ind]
fine_tune_points = [0, 7, 9, 13, 27, 29, 32, 34, 46, 48, 50, 59, 67, 69, 73, 75, 
                    82, 83, 90, 92, 96, 97, 98, 99, 103, 107, 108, 116, 118, 121, 131, 145, 
                    146, 150, 160, 163, 167, 170, 171, 179, 200, 207, 218, 220, 221, 224, 227, 228, 
                    229, 232, 237, 238, 242, 254, 256, 261, 273, 274, 275, 277, 278, 280, 285, 288, 
                    300, 304, 326, 327, 336, 348, 352, 356, 367, 371, 374, 377, 383, 391, 392, 393, 
                    394, 398, 409, 412, 414, 433, 436, 442, 463, 468, 473, 474, 477, 483, 499, 510, 
                    516, 522, 526, 546, 551, 554, 556, 563, 570, 571, 575, 579, 585, 596, 602, 603, 
                    608, 609, 615, 621, 623, 628, 638, 647, 648, 655, 659, 660, 661, 663, 668, 670, 
                    676, 682, 691, 697, 701, 703, 704, 706, 709, 719, 723, 726, 738, 740, 743, 751, 
                    752, 759, 763, 764, 767, 769, 773, 778, 782, 786, 791, 792, 796, 799, 816, 824, 
                    826, 829, 830, 831, 833, 846, 854, 859, 868, 872, 877, 880, 887, 892, 895, 900, 
                    904, 909, 913, 920, 923, 934, 945, 952, 953, 954, 956, 960, 962, 968, 972, 976, 
                    980, 981, 984, 988, 990, 991, 993, 999]
test_points = [i for i in remain if i not in fine_tune_points]

class PhaseClassifierBatch(nn.Module):
    def __init__(self, dim, num_bins=314):
        super().__init__()
        self.dim=dim
        self.num_bins=num_bins
        # Register bin centers for phase decoding
        k = torch.arange(num_bins).float()
        centers = -math.pi + (2*math.pi)*(k + 0.5)/num_bins
        self.register_buffer('bin_centers', centers)
        self.register_buffer('sin_c', centers.sin())
        self.register_buffer('cos_c', centers.cos())

    def forward(self, x, weights, temperature: float = 1.0):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        x = x * weights['ln_weight'] + weights['ln_bias']

        x = torch.bmm(weights['fc1_w'], x.unsqueeze(-1)).squeeze(-1) + weights['fc1_b']
        x = F.gelu(x)

        x = torch.bmm(weights['fc2_w'], x.unsqueeze(-1)).squeeze(-1) + weights['fc2_b']
        x = F.gelu(x)

        x = torch.bmm(weights['fc3_w'], x.unsqueeze(-1)).squeeze(-1) + weights['fc3_b']
        x = F.gelu(x)

        logits = torch.bmm(weights['out_w'], x.unsqueeze(-1)).squeeze(-1) + weights['out_b']
        probs = F.softmax(logits / temperature, dim=-1)

        s = torch.sum(probs * self.sin_c, dim=-1, keepdim=True)
        c = torch.sum(probs * self.cos_c, dim=-1, keepdim=True)
        phi_hat = torch.atan2(s, c)

        return phi_hat, logits

class TDBiasBatch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, td_idx, weights):
        return weights['td_bias']

class TransducerMultiHeadModelPhaseBatch(nn.Module):
    def __init__(self, num_fourier_freqs=16, num_bins=314):
        super().__init__()
        self.ff = FourierFeatures(6, num_frequencies=num_fourier_freqs)
        self.phase_head = PhaseClassifierBatch(197, num_bins=num_bins)
        self.td_bias = TDBiasBatch()

        self.register_buffer(
            "offset",
            torch.tensor([0.132, 0.132, 0.092])
        )

    def forward(self, target_xyz, trans_xyz, out_inter, in_inter, hu_line, weights):

        so_mm = target_xyz * 0.001 - self.offset
        td_mm = trans_xyz * 0.001 - self.offset

        ff_input = torch.cat([td_mm, so_mm], dim=-1)

        dist_out = torch.norm(out_inter - trans_xyz, dim=-1, keepdim=True) * 0.001
        dist_thru = torch.norm(out_inter - in_inter, dim=-1, keepdim=True) * 0.001
        dist_in = torch.norm(in_inter - target_xyz, dim=-1, keepdim=True) * 0.001

        tof = ((dist_out / 1500) + (dist_thru / 2262) + (dist_in / 1500)) * 1000

        enc_geo = self.ff(ff_input)

        dist_feat = torch.cat([dist_out, dist_thru, dist_in], dim=-1)
        hu_mean = hu_line.mean(dim=1, keepdim=True)

        geo_feat = torch.cat([enc_geo, dist_feat, tof, hu_mean], dim=-1)

        pred_phase, logits = self.phase_head(geo_feat, weights)

        return pred_phase, logits

class TransducerMultiHeadModelAmpBatch(nn.Module):
    def __init__(self, num_fourier_freqs=16):
        super().__init__()
        self.ff = FourierFeatures(7, num_frequencies=num_fourier_freqs)

        self.register_buffer(
            "offset",
            torch.tensor([0.132, 0.132, 0.092])
        )

    def forward(self, target_xyz, trans_xyz, out_inter, in_inter, hu_line, weights):
        so_mm = target_xyz * 0.001 - self.offset
        td_mm = trans_xyz * 0.001 - self.offset
        ff_input = torch.cat([td_mm, so_mm], dim=-1)

        out_inter = out_inter.float()
        in_inter = in_inter.float()

        dist_out = torch.norm(out_inter - trans_xyz, dim=-1, keepdim=True) * 0.001
        dist_thru = torch.norm(out_inter - in_inter, dim=-1, keepdim=True) * 0.001
        dist_in = torch.norm(in_inter - target_xyz, dim=-1, keepdim=True) * 0.001

        tof = ((dist_out / 1500) + (dist_thru / 2262) + (dist_in / 1500)) * 1000

        enc_geo = self.ff(ff_input)
        dist_feat = torch.cat([dist_out, dist_thru, dist_in], dim=-1)
        hu_mean = hu_line.mean(dim=1, keepdim=True)

        geo_feat = torch.cat([enc_geo, dist_feat, tof, hu_mean], dim=-1)

        x = torch.bmm(weights['fc1_w'], geo_feat.unsqueeze(-1)).squeeze(-1) + weights['fc1_b']
        x = F.gelu(x)
        x = torch.bmm(weights['fc2_w'], x.unsqueeze(-1)).squeeze(-1) + weights['fc2_b']
        x = F.gelu(x)
        x = torch.bmm(weights['fc3_w'], x.unsqueeze(-1)).squeeze(-1) + weights['fc3_b']
        x = F.gelu(x)
        x = torch.bmm(weights['fc4_w'], x.unsqueeze(-1)).squeeze(-1) + weights['fc4_b']
        return x

def load_phase_batch_weights(pth_paths, td_bias_paths=None, device='cuda'):
    weight_dict = {
        'ln_weight': [], 'ln_bias': [],
        'fc1_w': [], 'fc1_b': [],
        'fc2_w': [], 'fc2_b': [],
        'fc3_w': [], 'fc3_b': [],
        'out_w': [], 'out_b': [],
        'td_bias': [],
    }

    for i, path in enumerate(pth_paths):
        sd = torch.load(path, map_location='cpu')

        weight_dict['ln_weight'].append(sd['phase_head.in_ln.weight'])
        weight_dict['ln_bias'].append(sd['phase_head.in_ln.bias'])

        weight_dict['fc1_w'].append(sd['phase_head.mlp.0.weight'])
        weight_dict['fc1_b'].append(sd['phase_head.mlp.0.bias'])
        weight_dict['fc2_w'].append(sd['phase_head.mlp.3.weight'])
        weight_dict['fc2_b'].append(sd['phase_head.mlp.3.bias'])
        weight_dict['fc3_w'].append(sd['phase_head.mlp.6.weight'])
        weight_dict['fc3_b'].append(sd['phase_head.mlp.6.bias'])
        weight_dict['out_w'].append(sd['phase_head.out.weight'])
        weight_dict['out_b'].append(sd['phase_head.out.bias'])

        if 'td_bias.bias.weight' in sd:
            weight_dict['td_bias'].append(sd['td_bias.bias.weight'].squeeze(-1))
        else:
            weight_dict['td_bias'].append(torch.zeros(1)) 

    return {k: torch.stack(v).to(device) for k, v in weight_dict.items()}

def load_amp_batch_weights(pth_paths, device='cuda'):
    weight_dict = {
        'fc1_w': [], 'fc1_b': [],
        'fc2_w': [], 'fc2_b': [],
        'fc3_w': [], 'fc3_b': [],
        'fc4_w': [], 'fc4_b': [],
    }

    for path in pth_paths:
        sd = torch.load(path, map_location='cpu')
        weight_dict['fc1_w'].append(sd['amp_head.0.weight'])
        weight_dict['fc1_b'].append(sd['amp_head.0.bias'])
        weight_dict['fc2_w'].append(sd['amp_head.3.weight'])
        weight_dict['fc2_b'].append(sd['amp_head.3.bias'])
        weight_dict['fc3_w'].append(sd['amp_head.6.weight'])
        weight_dict['fc3_b'].append(sd['amp_head.6.bias'])
        weight_dict['fc4_w'].append(sd['amp_head.9.weight'])
        weight_dict['fc4_b'].append(sd['amp_head.9.bias'])

    return {k: torch.stack(v).to(device) for k, v in weight_dict.items()}

phase_paths = [f"D:/multielement/successful_codes/Seol/Final_Codes/Phase_model/area{aidx}/Fine_tuned/Skull8/{i+1:04d}.pth" for i in range(1024)]
phase_weights = load_phase_batch_weights(phase_paths, device=device)

amp_paths = [f"D:/multielement/successful_codes/Seol/Final_Codes/Amp_model/area{aidx}/Fine_tuned/Skull8/{i+1:04d}.pth" for i in range(1024)]
amp_weights = load_amp_batch_weights(amp_paths, device=device)

phase_model = TransducerMultiHeadModelPhaseBatch().to(device)
amp_model = TransducerMultiHeadModelAmpBatch().to(device)

vol_pre = skull[target_skull_idx].unsqueeze(0).expand(1024,-1,-1,-1) #[1, 264, 264, 164]
td_pre = td_vxl[target_skull_idx] #[1024,3]
td_pre = torch.round(td_pre)

phase_inference_log  = {'cmae':[], 
                        'circ':[], 
                        'huber':[]
                        }

amp_inference_log    = {'mae':[], 
                        'rel_engy_err':[], 
                        'huber':[]
                        }

time_log = {'phase_inf':[], 
            'amp_inf':[], 
            'wall_time':[]
            }

pred_vals = {'target_skull_idx':[], 
             'target_point_idx':[],
             'pred_phs':[],
             'pred_amp':[]
             }
so_pre_d = so_vxl[0][0].unsqueeze(0).expand(1024,-1) #[3]

_, entry_pre, exit_pre = extract_u_centered_cube(
    vol_pre, td_pre, so_pre_d,
    size_hwd=(20,20,128),
    step_hwd=(1.0,1.0,1.0),
    td_anchor=(10,10,2),
    align_corners=True
)

entry_pre = torch.round(entry_pre)
exit_pre = torch.round(exit_pre)
hu_line_pre = sample_hu_line(vol_pre, entry_pre, exit_pre, td_pre, so_pre_d)

dummy_so = so_pre_d.to(device)
dummy_td = td_pre.to(device)
dummy_entry = entry_pre.to(device)
dummy_exit = exit_pre.to(device)
dummy_hu = hu_line_pre.to(device)

with torch.no_grad():
    for _ in range(20):
        _ , _ = phase_model(dummy_so, dummy_td, dummy_entry, dummy_exit, dummy_hu, phase_weights)
        _ = amp_model(dummy_so, dummy_td, dummy_entry, dummy_exit, dummy_hu, amp_weights)

so_vxl = so_vxl.to(device)


with torch.no_grad():
    for target_point_idx in test_points:
        torch.cuda.synchronize()
        start_wall_time = time.time()
        so_pre = so_vxl[target_skull_idx][target_point_idx].unsqueeze(0).expand(1024,-1) #[3]
    
        _, entry_pre, exit_pre = extract_u_centered_cube(
            vol_pre, td_pre, so_pre,
            size_hwd=(20,20,128),
            step_hwd=(1.0,1.0,1.0),
            td_anchor=(10,10,2),
            align_corners=True
        )
    
        entry_pre = torch.round(entry_pre).to(device)
        exit_pre = torch.round(exit_pre).to(device)
        hu_line_pre = sample_hu_line(vol_pre, entry_pre, exit_pre, td_pre, so_pre).to(device)
        
        torch.cuda.synchronize()
        start_ph_time = time.time()
        pred_ph, pred_lgt = phase_model(
            so_pre,
            td_pre,
            entry_pre,
            exit_pre,
            hu_line_pre,
            phase_weights
        )
        torch.cuda.synchronize()
        end_ph_time = time.time()
    
        torch.cuda.synchronize()
        start_amp_time = time.time()
        pred_amp = amp_model(
            so_pre,
            td_pre,
            entry_pre,
            exit_pre,
            hu_line_pre,
            amp_weights
        )
        torch.cuda.synchronize()
        end_amp_time = time.time()

        torch.cuda.synchronize()
        end_wall_time = time.time()
    
        gt_ph = ph_list[target_skull_idx][target_point_idx]
        abs_diff = torch.atan2(torch.sin(pred_ph - gt_ph), torch.cos(pred_ph - gt_ph)).abs()
        ph_cmae = abs_diff.mean()
    
        ph_diff = torch.atan2(torch.sin(pred_ph - gt_ph), torch.cos(pred_ph - gt_ph))
        circ_loss = (1 - torch.cos(ph_diff)).mean()
    
        ph_huber = F.huber_loss(ph_diff, torch.zeros_like(ph_diff), delta=0.2)
    
        gt_amp = amp_list[target_skull_idx][target_point_idx]
        abs_diff = (gt_amp-pred_amp).abs()
        amp_mae = abs_diff.mean()
    
        E_pred = (pred_amp ** 2).sum()
        E_gt = (gt_amp ** 2).sum()
        rel_energy_error = torch.abs(E_pred - E_gt) / E_gt * 100  # [%]
    
        amp_diff = (gt_amp-pred_amp)
        amp_huber = F.huber_loss(amp_diff, torch.zeros_like(amp_diff), delta=0.2)
    
        ph_inf_time = end_ph_time - start_ph_time
        amp_inf_time = end_amp_time - start_amp_time
        wall_time = end_wall_time - start_wall_time
    
        # Save to Log
        # phase_inference_log['cmae'].append(ph_cmae)
        # phase_inference_log['circ'].append(circ_loss)
        # phase_inference_log['huber'].append(ph_huber)
    
        # amp_inference_log['mae'].append(amp_mae)
        # amp_inference_log['rel_engy_err'].append(rel_energy_error)
        # amp_inference_log['huber'].append(amp_huber)
    
        time_log['phase_inf'].append(ph_inf_time)
        time_log['amp_inf'].append(amp_inf_time)
        time_log['wall_time'].append(wall_time)
    
        # pred_vals['target_skull_idx'].append(target_skull_idx)
        # pred_vals['target_point_idx'].append(target_point_idx)
        # pred_vals['pred_phs'].append(pred_ph)
        # pred_vals['pred_amp'].append(pred_amp)
        # print(pred_amp)
    
        print('========================================================')
        print(f'Skull {target_skull_idx}, Target {target_point_idx}')
        # print('===========================')
        # print('Phase Pred Log')
        # print('---------------------------')
        # print(f'CMAE: {ph_cmae:.6f} rad')
        # print(f'CIRC: {circ_loss: .6f}')
        # print(f'HUBER: {ph_huber:.6f} rad')
        # print('===========================')
        # print('Amplitude Pred Log')
        # print('---------------------------')
        # print(f'MAE: {amp_mae:.6f} Pa')
        # print(f'REL ENERGY ERR: {rel_energy_error: .6f} %')
        # print(f'HUBER: {amp_huber:.6f} Pa')
        print('===========================')
        print(f'Phase inf time: {ph_inf_time:.6f} s')
        print(f'Amp inf time: {amp_inf_time: .6f} s')
        print(f'Wall clock time: {wall_time:.6f} s')
        print('========================================================')
        del pred_ph, pred_amp
        torch.cuda.empty_cache()
    
# torch.save(phase_inference_log,f'./Final_Codes/Ablation Study/AB3_Finetuning_pts/100 pts/area{aidx}/Phase/skull{target_skull_idx}_phase_inference_log.pt')
# torch.save(phase_inference_log,f'./Final_Codes/FINAL_PARALLEL_RESULTS/area{aidx}/skull{target_skull_idx}_phase_inference_log.pt')
# torch.save(amp_inference_log,f'./Final_Codes/FINAL_PARALLEL_RESULTS/area{aidx}/skull{target_skull_idx}_amp_inference_log.pt')
# torch.save(time_log,f'./Final_Codes/FINAL_PARALLEL_RESULTS/area{aidx}/skull{target_skull_idx}_time_log.pt')
# torch.save(pred_vals,f'./Final_Codes/FINAL_PARALLEL_RESULTS/area{aidx}/skull{target_skull_idx}_pred_vals.pt')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
import math
from defining_fcns import extract_u_centered_cube, make_circular_soft_label, sample_hu_line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_so = list(range(1000))
extra_ind = list([0,9,90,99,499,900,909,990,999])
remain = [i for i in all_so if i not in extra_ind]
# random.seed(42)
# fine_tune_points = extra_ind + random.sample(remain, 191)
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
val_points = [i for i in remain if i not in fine_tune_points]

aidx = 1

data = torch.load("D:/multielement/successful_codes/github upload/repo_example_data.pt")
skull = data['skull']
td_vxl = data['td_vxl']
so_vxl = data['so_vxl']
ph_list = data['ph_list']

nbr_td = 1024
num_bins = 314
bin_edges = torch.linspace(-np.pi, np.pi, num_bins + 1)

ph_list = ph_list.to(device)
bin_edges = bin_edges.to(device) 
target_class = torch.bucketize(ph_list, bin_edges, right=True) - 1
target_class = torch.clamp(target_class, 0, num_bins - 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

########################################################################################################
#                                                MODEL                                                 #
########################################################################################################
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2 ** torch.arange(0, num_frequencies).float()

    def forward(self, x):
        x = x.unsqueeze(-1) * self.freq_bands.to(x.device) * math.pi * 2
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(x.shape[0], -1)

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
    
class TransducerMultiHeadModel(nn.Module):
    def __init__(self, num_fourier_freqs=16, num_bins=314):
        super().__init__()
        self.ff = FourierFeatures(6, num_frequencies=num_fourier_freqs)
        self.phase_head = PhaseClassifier(197, num_bins=num_bins)
        
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
        pred_phase, logits = self.phase_head(geo_feat)

        return pred_phase, logits


########################################################################################################
#                                              FUNCTIONS                                               #
########################################################################################################
class MultiSkullDataset(torch.utils.data.Dataset):
    def __init__(self, skull, nbr_skull_list, so_vxl, td_vxl, ph_list, target_class, td_idx_list, so_idx_list):
        self.index_list = []  # (skull_idx, so_idx, td_idx)
        for skull_idx in nbr_skull_list:
            for so_idx in so_idx_list:
                for td_idx in td_idx_list:
                    self.index_list.append((skull_idx, so_idx, td_idx))
        self.skull = skull
        self.so_vxl = so_vxl
        self.td_vxl = td_vxl
        self.ph_list = ph_list
        self.target_class = target_class

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        skull_idx, so_idx, td_idx = self.index_list[idx]
        so = self.so_vxl[skull_idx, so_idx]
        td = self.td_vxl[skull_idx, td_idx]
        vol = self.skull[skull_idx].to(dtype=torch.float32)
        gt_phase = self.ph_list[skull_idx, so_idx, td_idx]
        soft_label = self.target_class[skull_idx, so_idx, td_idx]
        return skull_idx, so, td, gt_phase, soft_label, vol

########################################################################################################
#                                        TRAINING / VALIDATION                                         #
########################################################################################################
phase_loss_fn = nn.KLDivLoss(reduction='batchmean')
epochs = 50

for td_idx in range(1024):
    starting = time.time()
    print(f"Transducer index : {td_idx+1}")
    model = TransducerMultiHeadModel(num_fourier_freqs=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    dataset = MultiSkullDataset(skull=skull,
                                nbr_skull_list=[0],
                                so_vxl=so_vxl,
                                td_vxl=td_vxl,
                                ph_list=ph_list,
                                target_class=target_class,
                                td_idx_list=[td_idx],
                                so_idx_list=list(range(1000)))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=False)
    for epoch in range(epochs):
        model.train()
        total_loss, total_mae = 0, 0
        total_circ_loss = 0
        total_huber_loss = 0
        total_kl_loss = 0
        inf_time=0
        for skull_idx, so, td, gt_phase, soft_label, vol in train_loader:
            so = so.to(device)
            td = td.to(device)
            td = torch.round(td)
            gt_phase = gt_phase.to(device)
            soft_label = soft_label.to(device)
            vol = vol.to(device)
            with torch.no_grad():
                _, entry, exit_p = extract_u_centered_cube(vol, td, so, size_hwd=(20,20,128), step_hwd=(1.0,1.0,1.0), td_anchor=(10,10,2), align_corners=True)
                hu_data = sample_hu_line(vol, entry, exit_p, td, so, N=128, align_corners=True)
                hu_line = hu_data

            exit_p = torch.round(exit_p)
            entry = torch.round(entry)

            optimizer.zero_grad()
            
            start_time=time.time()
            preds, logits = model(so, td, entry, exit_p, hu_line, td_idx)
            end_time =time.time()

            log_probs = F.log_softmax(logits, dim=-1)
            soft_label = soft_label.float()
            soft_target = make_circular_soft_label(soft_label, num_bins=num_bins, smoothing_radius=2, sigma=0.5)

            diff = torch.atan2(torch.sin(preds - gt_phase), torch.cos(preds - gt_phase))
            mae = torch.mean(torch.abs(diff))
            circ_loss = 1 - torch.cos(diff)
            huber_loss = F.huber_loss(diff, torch.zeros_like(diff), delta=0.2)
            kl_loss = phase_loss_fn(log_probs, soft_target)
            loss =  kl_loss * 0.4 + circ_loss * 0.6
            loss = loss.mean()
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            mae = torch.mean(torch.abs(torch.atan2(torch.sin(preds - gt_phase), torch.cos(preds - gt_phase))))
            total_circ_loss += circ_loss.detach().mean().item()
            total_huber_loss += huber_loss.detach().item()
            total_kl_loss    += kl_loss.detach().item()
            total_loss       += loss.item()
            total_mae        += mae.item()
            inf_time += end_time - start_time
        print(f'{epoch+1}/{epochs} || KL Loss: {total_loss/len(train_loader):.6f} || Phase MAE: {total_mae/len(train_loader):.6f} || inf_time: {inf_time/len(train_loader):.6f}')

    # torch.save(model.state_dict(), f"./Final_Codes/Phase_model/area{aidx}/Base/{td_idx+1:04d}.pth")
    print(f'Base model {td_idx+1:04d} saved!')
    print(f'total pretraining time: {time.time() - starting} seconds')

    del train_loader, dataset, optimizer, model
    torch.cuda.empty_cache()

    import gc; gc.collect()

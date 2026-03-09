# -*- coding: utf-8 -*-
"""
Created on Wed / Sep 24 / 2025

@author: CMME Minju Seol
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

sk_idx = 0
aidx = 1

data = torch.load("data/repo_example_data.pt")
skull = data['skull']
td_vxl = data['td_vxl']
so_vxl = data['so_vxl']
amp_list = data['amp_list']

from defining_fcns import extract_u_centered_cube, sample_hu_line

########################################################################################################
#                                              FUNCTIONS                                               #
########################################################################################################
class MultiSkullDataset(torch.utils.data.Dataset):
    def __init__(self, skull, nbr_skull_list, so_vxl, td_vxl, amp_list, td_idx_list, so_idx_list):
        self.index_list = []  # (skull_idx, so_idx, td_idx)
        for skull_idx in nbr_skull_list:
            for so_idx in so_idx_list:
                for td_idx in td_idx_list:
                    self.index_list.append((skull_idx, so_idx, td_idx))
        self.skull = skull
        self.so_vxl = so_vxl
        self.td_vxl = td_vxl
        self.amp_list = amp_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        skull_idx, so_idx, td_idx = self.index_list[idx]
        so = self.so_vxl[skull_idx, so_idx]
        td = self.td_vxl[skull_idx, td_idx]
        vol = self.skull[skull_idx].to(dtype=torch.float32)
        gt_amp = self.amp_list[skull_idx, so_idx, td_idx]
        return skull_idx, so, td, gt_amp, vol

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

########################################################################################################
#                                               TRAINING                                               #
########################################################################################################
kl_loss = nn.KLDivLoss(reduction='batchmean')
loss_fn = nn.HuberLoss(delta=0.5)
fine_tune_epochs=30

for td_idx in range(1024):
    print(f"Transducer index : {td_idx+1}")
    ckpt_path = f'checkpoints/Amp_model/Base/{td_idx+1:04d}.pth'
    model = TransducerMultiHeadModelAmp(num_fourier_freqs=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    starting = time.time()
        
    ref_dataset = MultiSkullDataset(
        skull=skull, nbr_skull_list=[sk_idx], so_vxl=so_vxl, td_vxl=td_vxl,
        amp_list=amp_list, td_idx_list=[td_idx], so_idx_list=fine_tune_points
    )
    ref_loader = DataLoader(ref_dataset, batch_size=16, shuffle=True)
    for m in model.amp_head.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.15
    fine_tune_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(fine_tune_optimizer, mode='min', patience=10, factor=0.5)

    start_tuning = time.time()

    for fine_epoch in range(fine_tune_epochs):
        model.train()
        total_loss = total_mae = 0.0
        for skull_idx, so, td, gt_amp, vol in ref_loader:
            so, td, gt_amp, = so.to(device), td.to(device), gt_amp.to(device)
            td = torch.round(td)
            vol = vol.to(device)

            _, entry, exit_p = extract_u_centered_cube(vol, td, so, size_hwd=(20,20,128), step_hwd=(1.0,1.0,1.0), td_anchor=(10,10,2), align_corners=True)
            hu_data = sample_hu_line(vol, entry, exit_p, td, so, N=128, align_corners=True)
            hu_line = hu_data

            exit_p = torch.round(exit_p)
            entry = torch.round(entry)

            fine_tune_optimizer.zero_grad()
            pred_amp = model(so, td, entry, exit_p, hu_line, td_idx)

            loss = F.huber_loss(pred_amp, gt_amp)
            mae = torch.mean(torch.abs(gt_amp - pred_amp))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            fine_tune_optimizer.step()

            total_loss += loss.item()
            total_mae += mae.item()

        scheduler.step(total_loss/len(ref_loader))
        print(f'{fine_epoch+1}/{fine_tune_epochs} || Amp MAE (Pa): {total_mae/len(ref_loader):.6f}')
    end_tuning = time.time()

    model.eval()
    val_dataset = MultiSkullDataset(
        skull=skull, nbr_skull_list=[sk_idx], so_vxl=so_vxl, td_vxl=td_vxl,
        amp_list=amp_list,  td_idx_list=[td_idx], so_idx_list=val_points
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    A_ref = torch.max(torch.cat([amp_list[sk_idx][i] for i in val_points])).item()

    total_huber_val = 0.0
    total_mae_val = 0.0
    total_percent_val = 0.0
    start_time = time.time()
    with torch.no_grad():
        for skull_idx, so, td, gt_amp, vol in val_loader:
            so, td, gt_amp = so.to(device), td.to(device), gt_amp.to(device)
            td = torch.round(td)
            vol = vol.to(device)

            _, entry, exit_p = extract_u_centered_cube(vol, td, so, size_hwd=(20,20,128), step_hwd=(1.0,1.0,1.0), td_anchor=(10,10,2), align_corners=True)

            exit_p = torch.round(exit_p)
            entry = torch.round(entry)

            hu_data = sample_hu_line(vol, entry, exit_p, td, so, N=128, align_corners=True)
            hu_line = hu_data
    
            pred_amp = model(so, td, entry, exit_p, hu_line, td_idx)
            loss = F.huber_loss(pred_amp, gt_amp)
            mae_val = torch.mean(torch.abs(gt_amp - pred_amp))

            percent_error = torch.abs(pred_amp - gt_amp) / A_ref * 100
            mean_percent_error = percent_error.mean().item()

            total_huber_val += loss.item()
            total_mae_val += mae_val.item()
            total_percent_val += mean_percent_error
            
    # save_path = f'checkpoints/Amp_model/Fine_tuned/{td_idx+1:04d}.pth'
    # torch.save(model.state_dict(), save_path)

    inference_time = (time.time() - start_time) / len(val_loader)
    print(f'VALIDATION (fine-tuned) || Amp MAE (Pa): {total_mae_val/len(val_loader):.6f} || Amp MAE (%): {total_percent_val/len(val_loader):.6f}')
    print(f'Inference time: {inference_time:.6f}s')


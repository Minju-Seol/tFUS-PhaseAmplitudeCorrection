# -*- coding: utf-8 -*-
"""
@author: CMME Minju Seol
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Amplitude Regression Model
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

# Phase Classification Model
class TransducerMultiHeadModelPhase(nn.Module):
    def __init__(self, num_fourier_freqs=16, num_bins=314):
        super().__init__()
        self.ff = FourierFeatures(6, num_frequencies=num_fourier_freqs) 
        self.phase_head = PhaseClassifier(197, num_bins=num_bins)

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiSkullDatasetAmp(torch.utils.data.Dataset):
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

class MultiSkullDatasetPhase(torch.utils.data.Dataset):
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
    

def make_circular_soft_label(class_index, num_bins=314, smoothing_radius=1, sigma=0.5):
    B = class_index.size(0)
    device = class_index.device
    indices = torch.arange(num_bins, device=device).unsqueeze(0).expand(B, -1)  # [B, num_bins]
    class_index = class_index.view(-1).float()
    dist = (indices - class_index.unsqueeze(1)) % num_bins
    dist = torch.minimum(dist, num_bins - dist)
    mask = dist <= smoothing_radius
    weights = torch.exp(- (dist ** 2) / (2 * sigma ** 2)) * mask
    weights = weights / weights.sum(dim=1, keepdim=True)

    return weights

def extract_u_centered_cube(vol, td, so, size_hwd=(20,20,128), step_hwd=(1.0,1.0,1.0), td_anchor=(10,10,2), align_corners=True):
    vol = vol.unsqueeze(1)
    B, C, D, H, W = vol.shape
    dtype = vol.dtype
    device = vol.device

    td = td.to(device=device, dtype=dtype)
    so = so.to(device=device, dtype=dtype)
    
    v = so - td
    u = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)  # (B,3)
    with torch.no_grad():
        check_vals = []
        for b in range(len(td)):
            start = td[b]
            length = torch.norm(so[b] - td[b]).item()
            N = max(int(length / 1.0), 128)
            ts = torch.linspace(0, length, N, device=td.device)
            pts = start[None,:] + ts[:,None]*u[b][None,:]

            xg = 2.0*(pts[:,2]/(W-1.0)) - 1.0
            yg = 2.0*(pts[:,1]/(H-1.0)) - 1.0
            zg = 2.0*(pts[:,0]/(D-1.0)) - 1.0
            grid = torch.stack([xg,yg,zg],dim=-1).view(1,-1,1,1,3)
            grid = torch.clamp(grid, -1.0, 1.0)

            vals = F.grid_sample(vol[b:b+1], grid, mode='bilinear', align_corners=True)
            check_vals.append(vals.max().item())

    check_vals = torch.tensor(check_vals, device=td.device)
    flip_mask = check_vals < 0.05
    u[flip_mask] = -u[flip_mask]

    z_hat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
    proj = (u * z_hat).sum(dim=1, keepdim=True)
    n2 = z_hat - proj * u

    mask = torch.norm(n2, dim=1, keepdim=True) < 1e-6
    if mask.any():
        y_hat = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        proj_y = (u * y_hat).sum(dim=1, keepdim=True)
        n2 = torch.where(mask, y_hat - proj_y * u, n2)

    n2 = n2 / (torch.norm(n2, dim=1, keepdim=True) + 1e-8)
    n1 = torch.cross(u, n2, dim=-1)
    n1 = n1 / (torch.norm(n1, dim=1, keepdim=True) + 1e-8)

    sv, sw, su = [torch.tensor(v, dtype=dtype, device=device) for v in step_hwd]
    r0, c0, d0 = [torch.tensor(v, dtype=dtype, device=device) for v in td_anchor]

    p0 = td - r0*sv*n2 - c0*sw*n1 - d0*su*u 

    Hp, Wp, Dp = size_hwd
    ku = torch.arange(Dp, device=device, dtype=dtype) * su
    iv = torch.arange(Hp, device=device, dtype=dtype) * sv
    jw = torch.arange(Wp, device=device, dtype=dtype) * sw
    K, I, J = torch.meshgrid(ku, iv, jw, indexing='ij')
    base = torch.stack([K, I, J], dim=-1).to(dtype)

    coords = (p0[:,None,None,None,:]
              + base[None, ..., 0:1]*u[:,None,None,None,:]
              + base[None, ..., 1:2]*n2[:,None,None,None,:]
              + base[None, ..., 2:3]*n1[:,None,None,None,:])

    z = coords[...,0]
    y = coords[...,1]
    x = coords[...,2]

    if align_corners:
        xg = 2.0 * (x/(W-1.0)) - 1.0
        yg = 2.0 * (y/(H-1.0)) - 1.0
        zg = 2.0 * (z/(D-1.0)) - 1.0
    else:
        xg = 2.0*((x+0.5)/W) - 1.0  
        yg = 2.0*((y+0.5)/H) - 1.0  
        zg = 2.0*((z+0.5)/D) - 1.0  

    grid = torch.stack([xg, yg, zg], dim=-1)
    grid = torch.clamp(grid, -1.0, 1.0)     
    patch = F.grid_sample(vol, grid, mode='nearest', padding_mode='border',
                          align_corners=align_corners)   
    patch = patch[:,0].permute(0,2,3,1).contiguous()     
    
    def find_entry_exit(vol, td, so, u, step=1.0, threshold=0.5):

        B, _, D, H, W = vol.shape
        device = vol.device
        dtype = vol.dtype

        entry_pts = []
        exit_pts = []

        for b in range(B):

            start = td[b]
            end   = so[b]
            length = torch.norm(end - start).item()
            nsteps = int(length/step)
            nsteps = max(int(length / step), 2)

            ts = torch.linspace(0, length, nsteps, device=device, dtype=dtype)
            pts = start[None,:] + ts[:,None]*u[b][None,:]

            xg = 2.0*(pts[:,2]/(W-1.0)) - 1.0
            yg = 2.0*(pts[:,1]/(H-1.0)) - 1.0
            zg = 2.0*(pts[:,0]/(D-1.0)) - 1.0
            grid = torch.stack([xg,yg,zg],dim=-1)
            grid = grid.view(1, nsteps, 1, 1, 3)
            grid = torch.clamp(grid, -1.0, 1.0)

            vals = F.grid_sample(vol[b:b+1], grid, mode='nearest',
                                align_corners=True)
            vals = vals.view(-1)
            vals = torch.nan_to_num(vals)

            th = max(0.05, vals.mean() + 0.2*vals.std())
            mask = vals > th
            if mask.any():
                idx = torch.where(mask)[0]
                entry_idx = idx[0]
                exit_idx  = idx[-1]
                entry_pts.append(pts[entry_idx])
                exit_pts.append(pts[exit_idx])
            else:
                entry_pts.append(torch.full((3,), float('nan'), device=device, dtype=dtype))
                exit_pts.append(torch.full((3,), float('nan'), device=device, dtype=dtype))

        return torch.stack(entry_pts,dim=0), torch.stack(exit_pts,dim=0)

    entry, exit_p = find_entry_exit(vol, td, so, u, step=0.5, threshold=0.1)

    return patch, entry, exit_p 


def find_entry_exit(vol, td, so, u, step=1.0, threshold=0.5):

    B, _, D, H, W = vol.shape
    device = vol.device
    dtype = vol.dtype

    entry_pts = []
    exit_pts = []

    for b in range(B):

        start = td[b]
        end   = so[b]
        length = torch.norm(end - start).item()
        nsteps = int(length/step)
        nsteps = max(int(length / step), 2)

        ts = torch.linspace(0, length, nsteps, device=device, dtype=dtype)
        pts = start[None,:] + ts[:,None]*u[b][None,:]

        xg = 2.0*(pts[:,2]/(W-1.0)) - 1.0
        yg = 2.0*(pts[:,1]/(H-1.0)) - 1.0
        zg = 2.0*(pts[:,0]/(D-1.0)) - 1.0
        grid = torch.stack([xg,yg,zg],dim=-1)
        grid = grid.view(1, nsteps, 1, 1, 3)
        grid = torch.clamp(grid, -1.0, 1.0)

        vals = F.grid_sample(vol[b:b+1], grid, mode='nearest',
                            align_corners=True)
        vals = vals.view(-1)
        vals = torch.nan_to_num(vals)

        th = max(0.05, vals.mean() + 0.2*vals.std())
        mask = vals > th
        if mask.any():
            idx = torch.where(mask)[0]
            entry_idx = idx[0]
            exit_idx  = idx[-1]
            entry_pts.append(pts[entry_idx])
            exit_pts.append(pts[exit_idx])
        else:
            entry_pts.append(torch.full((3,), float('nan'), device=device, dtype=dtype))
            exit_pts.append(torch.full((3,), float('nan'), device=device, dtype=dtype))
        entry=torch.stack(entry_pts,dim=0)
        exit_p=torch.stack(exit_pts,dim=0)
    return entry, exit_p

@torch.no_grad()
def sample_hu_line(vol, entry, exit, td, so, N=128, align_corners=True):

    vol = vol.unsqueeze(1)
    B, _, D, H, W = vol.shape
    device, dtype = vol.device, vol.dtype

    valid = torch.isfinite(entry).all(dim=1) & torch.isfinite(exit).all(dim=1)

    start = torch.where(valid.unsqueeze(1), entry, so)
    end   = torch.where(valid.unsqueeze(1), exit,  td)

    v = end - start

    t = torch.linspace(0, 1, N, device=device, dtype=dtype)[None, :, None]
    pts = start[:, None, :] + t * v[:, None, :]                            

    x = pts[..., 2]; y = pts[..., 1]; z = pts[..., 0]
    if align_corners:
        gx = 2.0 * (x / (W - 1.0)) - 1.0
        gy = 2.0 * (y / (H - 1.0)) - 1.0
        gz = 2.0 * (z / (D - 1.0)) - 1.0
    else:
        gx = 2.0 * ((x + 0.5) / W) - 1.0
        gy = 2.0 * ((y + 0.5) / H) - 1.0
        gz = 2.0 * ((z + 0.5) / D) - 1.0

    grid = torch.stack([gx, gy, gz], dim=-1).view(B, N, 1, 1, 3) 

    vals = F.grid_sample(vol, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)                                                                     
    hu_line = vals.view(B, N)

    return hu_line

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2 ** torch.arange(0, num_frequencies).float().to(device)
        
    def forward(self, x):
        x = x.unsqueeze(-1) * self.freq_bands * math.pi * 2
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(x.shape[0], -1)


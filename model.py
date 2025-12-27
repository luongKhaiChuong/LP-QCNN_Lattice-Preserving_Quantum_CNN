import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import config

dev = qml.device("default.qubit", wires=config.NUM_WIRES)

def apply_hamiltonian_evolution(params, time, steps):
    dt = time / steps
    cols = int(np.sqrt(config.NUM_WIRES))
    jx_h, jy_h, jz_h, jx_v, jy_v, jz_v = [params[:, i] for i in range(6)]

    for _ in range(steps):
        for i in range(config.NUM_WIRES):
            if (i + 1) % cols != 0:
                qml.IsingXX(2 * jx_h * dt, wires=[i, i+1])
                qml.IsingYY(2 * jy_h * dt, wires=[i, i+1])
                qml.IsingZZ(2 * jz_h * dt, wires=[i, i+1])
        for i in range(config.NUM_WIRES):
            if i < cols * (cols - 1):
                qml.IsingXX(2 * jx_v * dt, wires=[i, i+cols])
                qml.IsingYY(2 * jy_v * dt, wires=[i, i+cols])
                qml.IsingZZ(2 * jz_v * dt, wires=[i, i+cols])

@qml.qnode(dev, interface='torch', diff_method='backprop')
def aqc_kernel_parallel(inputs, filter_params):
    for j in range(config.NUM_WIRES):
        qml.RY(inputs[:, j] * np.pi, wires=j)
    apply_hamiltonian_evolution(filter_params, config.TIME_DURATION, config.TIME_STEPS)
    return [qml.expval(qml.PauliZ(i)) for i in range(config.NUM_WIRES)]

class SimpleQuantumCNN(nn.Module):
    def __init__(self, n_filters=config.DEFAULT_NUM_FILTERS, n_classes=10, in_channels=1):
        super().__init__()
        self.n_filters = n_filters
        self.in_channels = in_channels
        self.kernel_size = int(np.sqrt(config.NUM_WIRES))
        self.stride = config.STRIDE
        
        if self.in_channels > 1:
            if self.n_filters % self.in_channels != 0:
                raise ValueError(f"Filters ({n_filters}) must be divisible by in_channels ({in_channels})")
            self.filters_per_channel = self.n_filters // self.in_channels
        else:
            self.filters_per_channel = self.n_filters

        self.hamiltonian_params = nn.Parameter(torch.randn(self.n_filters, 6))
        with torch.no_grad():
             for k in range(self.n_filters):
                 self.hamiltonian_params[k, k % 6] += 0.2
        self.alpha = 0.5 
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(n_classes))

    def forward(self, x):
        batch_size = x.shape[0]
        if self.in_channels == 1:
            patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
            n_patches = patches.shape[2]
            input_flat = patches.transpose(1, 2).reshape(-1, config.NUM_WIRES)
            input_scaled = input_flat * self.alpha
            inputs_expanded = input_scaled.repeat_interleave(self.n_filters, dim=0)
            params_expanded = self.hamiltonian_params.unsqueeze(0).expand(input_scaled.shape[0], -1, -1).reshape(-1, 6)
        else:
            input_parts = []
            params_list = []
            n_patches = None
            for c in range(self.in_channels):
                x_c = x[:, c:c+1, :, :]
                patches_c = F.unfold(x_c, kernel_size=self.kernel_size, stride=self.stride)
                if n_patches is None: n_patches = patches_c.shape[2]
                flat_c = patches_c.transpose(1, 2).reshape(-1, config.NUM_WIRES)
                scaled_c = flat_c * self.alpha
                expanded_c = scaled_c.repeat_interleave(self.filters_per_channel, dim=0)
                input_parts.append(expanded_c)
                start_idx = c * self.filters_per_channel
                end_idx = start_idx + self.filters_per_channel
                subset_params = self.hamiltonian_params[start_idx:end_idx]
                p_exp = subset_params.unsqueeze(0).expand(batch_size * n_patches, -1, -1).reshape(-1, 6)
                params_list.append(p_exp)
            inputs_expanded = torch.cat(input_parts, dim=0)
            params_expanded = torch.cat(params_list, dim=0)

        q_out_list = aqc_kernel_parallel(inputs_expanded, params_expanded)
        q_out = torch.stack(q_out_list, dim=1).float()
        
        if self.in_channels == 1:
             q_out_reshaped = q_out.view(batch_size * n_patches, self.n_filters, config.NUM_WIRES)
             combined_features = q_out_reshaped.reshape(batch_size * n_patches, -1)
             total_channels = self.n_filters * config.NUM_WIRES
        else:
             chunk_size = q_out.shape[0] // self.in_channels
             chunks = torch.split(q_out, chunk_size, dim=0)
             rec_maps = []
             for chunk in chunks:
                 chunk_r = chunk.view(batch_size * n_patches, self.filters_per_channel, config.NUM_WIRES)
                 chunk_flat = chunk_r.reshape(batch_size * n_patches, -1)
                 rec_maps.append(chunk_flat)
             combined_features = torch.cat(rec_maps, dim=1)
             total_channels = self.n_filters * config.NUM_WIRES

        side_len = int(np.sqrt(n_patches))
        x_map = combined_features.view(batch_size, n_patches, total_channels).permute(0, 2, 1).view(batch_size, total_channels, side_len, side_len)
        return self.net(x_map)
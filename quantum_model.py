import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import config

# Khởi tạo device cho Pennylane
# Lưu ý: Trên HPCC nếu có GPU xịn, hãy cài 'pennylane-lightning[gpu]' để chạy nhanh hơn
dev = qml.device("default.qubit", wires=config.N_WIRES)

def build_hamiltonian_nxn(params):
    jx_h, jy_h, jz_h, jx_v, jy_v, jz_v = params
    coeffs, ops = [], []
    cols = int(np.sqrt(config.N_WIRES))
    
    for i in range(config.N_WIRES):
        # Horizontal
        if (i + 1) % cols != 0: 
            ops.append(qml.IsingXX(wires=[i, i+1])); coeffs.append(jx_h)
            ops.append(qml.IsingYY(wires=[i, i+1])); coeffs.append(jy_h)
            ops.append(qml.IsingZZ(wires=[i, i+1])); coeffs.append(jz_h)  
        # Vertical
        if i < cols * (cols - 1): 
            ops.append(qml.IsingXX(wires=[i, i+cols])); coeffs.append(jx_v)
            ops.append(qml.IsingYY(wires=[i, i+cols])); coeffs.append(jy_v)
            ops.append(qml.IsingZZ(wires=[i, i+cols])); coeffs.append(jz_v)  
            
    return qml.Hamiltonian(coeffs, ops)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def aqc_kernel_backprop(inputs, layer_params):
    for j in range(config.N_WIRES):
        qml.RY(inputs[:, j] * np.pi, wires=j)
    
    for l in range(config.N_LAYERS):
        H = build_hamiltonian_nxn(layer_params[l])
        qml.ApproxTimeEvolution(H, config.TIME, config.TIME_STEPS)

    return [qml.expval(qml.PauliZ(i)) for i in range(config.N_WIRES)]

class AQCBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = int(np.sqrt(config.N_WIRES))
        self.stride = config.STRIDE
        
        self.hamiltonian_params = nn.Parameter(torch.rand(config.N_LAYERS, 6) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(0.25))

        self.total_channels = config.N_WIRES 
        flat_dim = config.FIN_KERNEL[0] * config.FIN_KERNEL[1] * self.total_channels
        
        self.net = nn.Sequential(
            nn.BatchNorm1d(flat_dim),
            nn.Linear(flat_dim, 2)
        )

    def forward(self, x):
        b_size, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        n_patches = patches.shape[-1]
        
        x_flat = patches.transpose(1, 2).reshape(-1, config.N_WIRES)
        x_scaled = x_flat * self.alpha 

        q_out_list = aqc_kernel_backprop(x_scaled, self.hamiltonian_params)
        q_out = torch.stack(q_out_list, dim=1).float()
        
        side_len = int(np.sqrt(n_patches))
        x_map = q_out.view(b_size, side_len, side_len, self.total_channels).permute(0, 3, 1, 2)
        x_pooled = F.adaptive_avg_pool2d(x_map, config.FIN_KERNEL)
        return self.net(x_pooled.flatten(1))
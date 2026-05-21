import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
import config
from utils import generate_grid_topology

class LPQCNN(nn.Module):
    def __init__(self, n_filters, time_dyn, time_steps, n_classes=10, kernel_size=2, stride=1, noise_prob=0.0): # <-- Thêm noise_prob
        super().__init__()
        self.n_filters = n_filters
        self.time_dyn = time_dyn
        self.time_steps = time_steps
        self.n_classes = n_classes 
        self.kernel_size = kernel_size
        self.stride = stride
        self.alpha = 0.5 
        self.noise_prob = noise_prob
        
        self.n_wires = config.N_WIRES
        self.h_pairs, self.v_pairs = generate_grid_topology(config.GRID_H, config.GRID_W)
        self.hamiltonian_params = nn.Parameter(torch.randn(n_filters, 6))
        
        if self.noise_prob > 0.0:
            self.dev = qml.device("default.mixed", wires=self.n_wires)
        else:
            self.dev = qml.device("default.qubit", wires=self.n_wires)
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, params):
            # Kiểm tra xem inputs có phải là batch (2D) hay mẫu đơn (1D)
            is_batched = (inputs.ndim == 2)
            
            for j in range(self.n_wires): 
                # Nếu là batch, lấy toàn bộ cột j. Nếu là mẫu đơn, lấy phần tử j.
                val = inputs[:, j] if is_batched else inputs[j]
                qml.RY(val * np.pi, wires=j)
            
            dt = self.time_dyn / self.time_steps
            jx_h, jy_h, jz_h, jx_v, jy_v, jz_v = params
            
            for _ in range(self.time_steps):
                for w in self.h_pairs:
                    qml.IsingXX(2*jx_h*dt, wires=w); qml.IsingYY(2*jy_h*dt, wires=w); qml.IsingZZ(2*jz_h*dt, wires=w)
                for w in self.v_pairs:
                    qml.IsingXX(2*jx_v*dt, wires=w); qml.IsingYY(2*jy_v*dt, wires=w); qml.IsingZZ(2*jz_v*dt, wires=w)
                
                # THÊM NHIỄU
                if getattr(self, 'noise_prob', 0.0) > 0.0:
                    for w in range(self.n_wires):
                        qml.DepolarizingChannel(self.noise_prob, wires=w)
                        
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        
        self.qnode = circuit
        self.vmap_qnode = torch.vmap(self.qnode, in_dims=(0, None))

        dim = 14
        out_dim = (dim - kernel_size) // stride + 1
        initial_input = n_filters * self.n_wires * out_dim * out_dim
        self.fc = nn.Linear(initial_input, n_classes)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride) 
        num_patches = patches.shape[-1]
        x_flat = patches.transpose(1, 2).reshape(-1, self.n_wires) * self.alpha 
        
        outputs = []
        for k in range(self.n_filters):
            p = self.hamiltonian_params[k]
            res = self.qnode(x_flat, p)
            
            if isinstance(res, (list, tuple)):
                res = torch.stack(res, dim=-1)
            outputs.append(res)
            
        out_tensor = torch.stack(outputs, dim=1) 
        out_tensor = out_tensor.view(batch_size, num_patches, self.n_filters, self.n_wires)
        
        return self.fc(out_tensor.reshape(batch_size, -1).float())
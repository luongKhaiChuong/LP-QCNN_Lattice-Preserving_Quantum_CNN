import torch
import pennylane as qml
import functools
import config

class Analyzer:
    @staticmethod
    def calculate_bch_error(model):
        dev_torch = next(model.parameters()).device
        sys_dim = 2 ** model.n_wires
        I = torch.eye(2, device=dev_torch, dtype=torch.complex64)
        X = torch.tensor([[0,1],[1,0]], device=dev_torch, dtype=torch.complex64)
        Y = torch.tensor([[0,-1j],[1j,0]], device=dev_torch, dtype=torch.complex64)
        Z = torch.tensor([[1,0],[0,-1]], device=dev_torch, dtype=torch.complex64)
        
        def k_op(m1, m2, w, n_wires): 
            ops = [I] * n_wires
            ops[w[0]] = m1; ops[w[1]] = m2
            return functools.reduce(torch.kron, ops)
        
        errors = []
        with torch.no_grad():
            for i in range(model.n_filters):
                p = model.hamiltonian_params[i]
                He = torch.zeros((sys_dim, sys_dim), device=dev_torch, dtype=torch.complex64)
                Ho = torch.zeros((sys_dim, sys_dim), device=dev_torch, dtype=torch.complex64)
                for w in model.h_pairs: 
                    He += p[0]*k_op(X,X,w, model.n_wires) + p[1]*k_op(Y,Y,w, model.n_wires) + p[2]*k_op(Z,Z,w, model.n_wires)
                for w in model.v_pairs: 
                    Ho += p[3]*k_op(X,X,w, model.n_wires) + p[4]*k_op(Y,Y,w, model.n_wires) + p[5]*k_op(Z,Z,w, model.n_wires)
                
                comm = torch.matmul(He, Ho) - torch.matmul(Ho, He)
                err_val = ((model.time_dyn**2)/(model.time_steps)) * torch.linalg.norm(comm, 2).item()
                errors.append(err_val)
        return torch.tensor(errors).mean().item()

    @staticmethod
    def calculate_fisher_rank(model, loader):
        try:
            inputs, _ = next(iter(loader))
            inputs = inputs[:8].to(config.DEVICE)
            dev = qml.device("default.qubit", wires=model.n_wires)
            
            @qml.qnode(dev, interface="torch")
            def qnode_state(x, p, t, s):
                for j in range(model.n_wires): qml.RY(x[j]*np.pi, wires=j)
                dt = t/s
                jx_h,jy_h,jz_h,jx_v,jy_v,jz_v = p
                for _ in range(s):
                    for w in model.h_pairs:
                        qml.IsingXX(2*jx_h*dt, wires=w); qml.IsingYY(2*jy_h*dt, wires=w); qml.IsingZZ(2*jz_h*dt, wires=w)
                    for w in model.v_pairs:
                        qml.IsingXX(2*jx_v*dt, wires=w); qml.IsingYY(2*jy_v*dt, wires=w); qml.IsingZZ(2*jz_v*dt, wires=w)
                return qml.state()

            patches = torch.nn.functional.unfold(inputs, kernel_size=model.kernel_size, stride=model.stride)
            x_flat = patches.transpose(1, 2).reshape(-1, model.n_wires) * model.alpha
            x_sample = x_flat[torch.randperm(x_flat.size(0))[:8]]
            
            ranks = []
            for k in range(model.n_filters):
                params = model.hamiltonian_params[k]
                QFIM = torch.zeros((6, 6), device=config.DEVICE, dtype=torch.float32)
                for x in x_sample:
                    jac_real = torch.autograd.functional.jacobian(lambda p: qnode_state(x, p, model.time_dyn, model.time_steps).real, params)
                    jac_imag = torch.autograd.functional.jacobian(lambda p: qnode_state(x, p, model.time_dyn, model.time_steps).imag, params)
                    jac = (jac_real + 1j * jac_imag).to(dtype=torch.complex64)
                    psi = qnode_state(x, params, model.time_dyn, model.time_steps).detach().to(config.DEVICE).to(dtype=torch.complex64)
                    J_dag = jac.T.conj()
                    term1 = torch.matmul(J_dag, jac) 
                    term2 = torch.matmul(J_dag, psi.unsqueeze(1)) @ torch.matmul(psi.unsqueeze(0), jac)
                    QFIM += 4 * (term1 - term2).real
                QFIM /= len(x_sample)
                S = torch.linalg.svdvals(QFIM)
                S = S / S.sum()
                S = S[S > 1e-6]
                eff_rank = torch.exp(-torch.sum(S * torch.log(S))).item()
                ranks.append(eff_rank)
            return np.mean(ranks)
        except Exception as e:
            print(f"Fisher calc failed: {e}")
            return 0.0
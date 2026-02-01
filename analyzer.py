import torch
import pennylane as qml
from pennylane import numpy as np 
import functools
import config
import logging

class Analyzer:
    """
    A static utility class for analyzing the physical fidelity and geometric 
    properties of Lattice-Preserving Quantum Convolutional Neural Networks (LP-QCNN).
    
    This class provides metrics to evaluate:
    1. Simulation Accuracy: Via Baker-Campbell-Hausdorff (BCH) Error.
    2. Model Expressivity: Via the Effective Rank of the Quantum Fisher Information Matrix (QFIM).
    """

    @staticmethod
    def calculate_bch_error(qnn_model):
        """
        Calculates the Baker-Campbell-Hausdorff (BCH) Error to quantify the 
        Trotterization approximation error in the quantum circuit.

        ### Theoretical Background
        The model simulates the time-evolution of a Hamiltonian $H = H_{hor} + H_{ver}$.
        Since the horizontal interactions ($H_{hor}$) and vertical interactions ($H_{ver}$) 
        do not commute (i.e., $[H_{hor}, H_{ver}] \neq 0$), approximating the evolution 
        $e^{-iH t}$ using alternating layers of gates (Trotterization) introduces an error.

        ### Formula
        The second-order Trotter error is proportional to the norm of the commutator:
        $$ \mathcal{E} \approx \frac{t^2}{S} \left\| [H_{hor}, H_{ver}] \right\|_F $$
        
        Where:
        - $t$: Total evolution time (`time_dyn`).
        - $S$: Number of Trotter steps (`time_steps`).
        - $[A, B] = AB - BA$: The commutator.
        - $\| \cdot \|_F$: The spectral norm.
        - Up to a constant prefactor (theoretical second-order Trotter), the error scales as \frac{t^2}{S}
        Args:
            qnn_model (LPQCNN): The quantum model instance containing Hamiltonian parameters
                                and topology definitions (h_pairs, v_pairs).

        Returns:
            float: The mean BCH error averaged over all filters. A lower value indicates 
                   a more physically accurate simulation.
        """
        # Retrieve the device (CPU/GPU) to ensure tensors are created on the same device
        device = next(qnn_model.parameters()).device
        
        # Calculate system dimension: 2^(number of qubits)
        system_dim = 2 ** qnn_model.n_wires
        
        # Define basic Pauli matrices (Complex-valued)
        I = torch.eye(2, device=device, dtype=torch.complex64)
        X = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.complex64)
        Y = torch.tensor([[0, -1j], [1j, 0]], device=device, dtype=torch.complex64)
        Z = torch.tensor([[1, 0], [0, -1]], device=device, dtype=torch.complex64)
        
        def compute_kron_product(op1, op2, target_wires, num_wires):
            """
            Helper function to construct an n-qubit operator via Kronecker product.
            Constructs: I ⊗ ... ⊗ op1 ⊗ ... ⊗ op2 ⊗ ... ⊗ I
            """
            ops_sequence = [I] * num_wires
            ops_sequence[target_wires[0]] = op1
            ops_sequence[target_wires[1]] = op2
            return functools.reduce(torch.kron, ops_sequence)
        
        bch_errors = []
        
        # Disable gradient calculation for efficiency (Analysis mode only)
        with torch.no_grad():
            for filter_idx in range(qnn_model.n_filters):
                # Extract parameters for the current filter
                # params shape: [6] -> [jx_h, jy_h, jz_h, jx_v, jy_v, jz_v]
                params = qnn_model.hamiltonian_params[filter_idx]
                
                # Initialize Horizontal (H_hor) and Vertical (H_ver) Hamiltonians
                H_horizontal = torch.zeros((system_dim, system_dim), device=device, dtype=torch.complex64)
                H_vertical = torch.zeros((system_dim, system_dim), device=device, dtype=torch.complex64)
                
                # --- Construct H_horizontal ---
                # Sum of interactions on all horizontal edges
                for wire_pair in qnn_model.h_pairs: 
                    term_x = params[0] * compute_kron_product(X, X, wire_pair, qnn_model.n_wires)
                    term_y = params[1] * compute_kron_product(Y, Y, wire_pair, qnn_model.n_wires)
                    term_z = params[2] * compute_kron_product(Z, Z, wire_pair, qnn_model.n_wires)
                    H_horizontal += (term_x + term_y + term_z)
                
                # --- Construct H_vertical ---
                # Sum of interactions on all vertical edges
                for wire_pair in qnn_model.v_pairs: 
                    term_x = params[3] * compute_kron_product(X, X, wire_pair, qnn_model.n_wires)
                    term_y = params[4] * compute_kron_product(Y, Y, wire_pair, qnn_model.n_wires)
                    term_z = params[5] * compute_kron_product(Z, Z, wire_pair, qnn_model.n_wires)
                    H_vertical += (term_x + term_y + term_z)
                
                # --- Compute Commutator ---
                # [H_hor, H_ver] = H_hor * H_ver - H_ver * H_hor
                commutator = torch.matmul(H_horizontal, H_vertical) - torch.matmul(H_vertical, H_horizontal)
                
                # --- Compute Scaled Error ---
                scaling_factor = (qnn_model.time_dyn ** 2) / qnn_model.time_steps
                error_magnitude = scaling_factor * torch.linalg.norm(commutator, 2).item() # Spectral norm
                
                bch_errors.append(error_magnitude)
                
        return torch.tensor(bch_errors).mean().item()

    @staticmethod
    def calculate_fisher_rank(qnn_model, data_loader):
        """
        Calculates the Effective Rank of the Quantum Fisher Information Matrix (QFIM).

        ### Theoretical Background
        The QFIM metric characterizes the geometry of the parameter space. The Effective Rank
        indicates the dimensionality of the "useful" parameter space.
        - **High Rank:** The model parameters are independent and expressive (good trainability).
        - **Low Rank:** The model suffers from over-parameterization or Barren Plateaus (poor trainability).
        Since the circuit outputs a pure quantum state ∣ψ(θ)⟩, we employ the pure-state Quantum Fisher Information Matrix.
        ### Formula
        1. **QFIM Elements ($F_{ij}$):**
           $$ F_{ij} = 4 \cdot \text{Re} \left[ \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle \right] $$
           
        2. **Effective Rank ($R_{eff}$):**
           Calculated using the Shannon entropy of the normalized singular values ($\tilde{\sigma}_k$):
           $$ R_{eff} = \exp \left( - \sum_k \tilde{\sigma}_k \ln \tilde{\sigma}_k \right) $$

        Args:
            qnn_model (LPQCNN): The quantum model to analyze.
            data_loader (DataLoader): The data provider.

        Returns:
            float: The mean effective rank calculated over a small batch of data samples.
        Note: We suppose 8 samples are sufficient since the QFIM characterizes local geometry of the parameter space and is used 
        here as a qualitative diagnostic rather than a precise estimator. A low effective rank indicates that the parameter space collapses onto a low-dimensional manifold, a known signature associated with barren plateau phenomena.
        """
        try:
            # Load a small batch (8 samples) to reduce computational cost
            inputs, _ = next(iter(data_loader))
            inputs = inputs[:8].to(config.DEVICE)
            
            # Create a temporary PennyLane device for state vector simulation
            dev_pennylane = qml.device("default.qubit", wires=qnn_model.n_wires)
            
            # Define the QNode: Input -> Quantum State Vector |psi>
            @qml.qnode(dev_pennylane, interface="torch")
            def get_quantum_state(input_data, params, total_time, steps):
                # Encoding Layer
                for j in range(qnn_model.n_wires): 
                    qml.RY(input_data[j] * np.pi, wires=j)
                
                dt = total_time / steps
                # Unpack parameters: 3 for Horizontal, 3 for Vertical
                jx_h, jy_h, jz_h, jx_v, jy_v, jz_v = params
                
                # Trotterization Loop
                for _ in range(steps):
                    # Horizontal Evolution
                    for wire_pair in qnn_model.h_pairs:
                        qml.IsingXX(2*jx_h*dt, wires=wire_pair)
                        qml.IsingYY(2*jy_h*dt, wires=wire_pair)
                        qml.IsingZZ(2*jz_h*dt, wires=wire_pair)
                    # Vertical Evolution
                    for wire_pair in qnn_model.v_pairs:
                        qml.IsingXX(2*jx_v*dt, wires=wire_pair)
                        qml.IsingYY(2*jy_v*dt, wires=wire_pair)
                        qml.IsingZZ(2*jz_v*dt, wires=wire_pair)
                
                return qml.state()

            # Preprocess inputs: Unfold images into patches
            patches = torch.nn.functional.unfold(inputs, kernel_size=qnn_model.kernel_size, stride=qnn_model.stride)
            x_flat = patches.transpose(1, 2).reshape(-1, qnn_model.n_wires) * qnn_model.alpha
            
            # Randomly sample 8 patches from the image batch
            x_sample = x_flat[torch.randperm(x_flat.size(0))[:8]]
            
            ranks = []
            
            # Iterate through each filter to compute its specific Fisher Rank
            for k in range(qnn_model.n_filters):
                current_params = qnn_model.hamiltonian_params[k]
                
                # Initialize QFIM (6 parameters -> 6x6 matrix)
                fisher_matrix = torch.zeros((6, 6), device=config.DEVICE, dtype=torch.float32)
                
                for x_val in x_sample:
                    # 1. Compute Jacobian (Derivative of state |psi> w.r.t params)
                    # Note: PyTorch Jacobian splits complex numbers into Real and Imaginary parts
                    jac_real = torch.autograd.functional.jacobian(
                        lambda p: get_quantum_state(x_val, p, qnn_model.time_dyn, qnn_model.time_steps).real, 
                        current_params
                    )
                    jac_imag = torch.autograd.functional.jacobian(
                        lambda p: get_quantum_state(x_val, p, qnn_model.time_dyn, qnn_model.time_steps).imag, 
                        current_params
                    )
                    jacobian = (jac_real + 1j * jac_imag).to(dtype=torch.complex64)
                    
                    # 2. Get the State Vector |psi>
                    state_vector = get_quantum_state(
                        x_val, current_params, qnn_model.time_dyn, qnn_model.time_steps
                    ).detach().to(config.DEVICE).to(dtype=torch.complex64)
                    
                    # 3. Calculate terms for F_ij = <di|dj> - <di|psi><psi|dj>
                    jac_conjugate_transpose = jacobian.T.conj()
                    
                    # Term 1: Overlap of derivatives (<di|dj>)
                    overlap_derivatives = torch.matmul(jac_conjugate_transpose, jacobian) 
                    
                    # Term 2: Correction (<di|psi><psi|dj>)
                    correction_term = torch.matmul(jac_conjugate_transpose, state_vector.unsqueeze(1)) @ \
                                      torch.matmul(state_vector.unsqueeze(0), jacobian)
                    
                    # Accumulate Real part (Fisher Information is real)
                    fisher_matrix += 4 * (overlap_derivatives - correction_term).real
                
                # Average QFIM over the batch
                fisher_matrix /= len(x_sample)
                
                # 4. Compute Effective Rank via Singular Value Decomposition (SVD)
                singular_values = torch.linalg.svdvals(fisher_matrix)
                
                # Normalize singular values to treat them as probabilities
                normalized_sv = singular_values / singular_values.sum()
                
                # Filter out negligible values to avoid log(0)
                normalized_sv = normalized_sv[normalized_sv > 1e-6] 
                
                # Entropy-based Effective Rank
                entropy = -torch.sum(normalized_sv * torch.log(normalized_sv))
                eff_rank = torch.exp(entropy).item()
                
                ranks.append(eff_rank)
                
            return np.mean(ranks)
            
        except Exception as e:
            error_msg = f"Fisher Rank calculation failed: {str(e)}"
            if config.LOGGER:
                config.LOGGER.error(error_msg)
            else:
                print(error_msg)
            return 0.0
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentODE(nn.Module):
    '''
    Velocity-Latent Linear ODE model for 3D Multi-Object Tracking. This module shifts the
    paradigm from learning absolute coordinates to learning bounded velocity (increments),
    perfectly aligning with the norm-preserving nature of skew-symmetric matrices to resolve the
    "orbit collapse" paradox.
    '''

    def __init__(self, physical_dim=7, latent_dyn_dim=32, latent_stat_dim=16,
                 control_dim=6, input_mode='velocity'):
        '''Initialize the object state and configure the resources required by this component.'''
        super(LatentODE, self).__init__()

        self.physical_dim = physical_dim
        self.latent_dyn_dim = latent_dyn_dim
        self.latent_stat_dim = latent_stat_dim
        self.latent_dim = latent_dyn_dim + latent_stat_dim
        self.control_dim = control_dim
        self.input_mode = input_mode

        # In position mode, encoder receives [x_curr, x_prev] = 14 dims
        # to eliminate finite-difference noise amplification (Path 3)
        enc_input_dim = 2 * physical_dim if input_mode == 'position' else physical_dim

        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )

        self.shared_trunk = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
        )

        self.motion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.motion_unc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.size_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.size_unc_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # P7: bounded residual adapter.  The final residual and gate layers are
        # initialized so that a newly added adapter is exactly an identity path
        # for legacy/UARO checkpoints until it is explicitly enabled and trained.
        self.adapter_context_dim = 21
        self.residual_adapter = nn.Sequential(
            nn.Linear(self.adapter_context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        self.adapter_gate = nn.Sequential(
            nn.Linear(self.adapter_context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)
        nn.init.zeros_(self.adapter_gate[-1].weight)
        nn.init.constant_(self.adapter_gate[-1].bias, -4.0)
        self.register_buffer(
            'adapter_delta_max',
            torch.tensor([0.5, 0.5, 0.5, 0.15], dtype=torch.float32),
        )

        self.v_slip_threshold = nn.Parameter(torch.tensor([0.1]))

        self.trans_dim = latent_dyn_dim // 2
        self.rot_dim = latent_dyn_dim - self.trans_dim

        self.A_trans_params = nn.Parameter(torch.randn(self.trans_dim, self.trans_dim) * 0.01)
        self.A_rot_params = nn.Parameter(torch.randn(self.rot_dim, self.rot_dim) * 0.01)

        self.A_trans_diag_raw = nn.Parameter(torch.zeros(self.trans_dim))
        self.A_rot_diag_raw = nn.Parameter(torch.zeros(self.rot_dim))

        self.lambda_param = nn.Parameter(torch.tensor([0.1]))
        self.depth_norm = nn.Parameter(torch.tensor([30.0]))

        self.phi_net = nn.Sequential(
            nn.Linear(self.latent_dyn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dyn_dim)
        )

        self.nonlin_net = nn.Sequential(
            nn.Linear(self.latent_dyn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dyn_dim)
        )

        self.epsilon = nn.Parameter(torch.tensor([0.01]))
        self.epsilon_nonlin = nn.Parameter(torch.tensor([0.01]))

        self.H = nn.Parameter(torch.eye(self.latent_dyn_dim))
        self.B = nn.Parameter(torch.eye(self.latent_dyn_dim))

        # --- CDE components ---

        self.cde_field = nn.Sequential(
            nn.Linear(latent_dyn_dim + control_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dyn_dim),
        )

        nn.init.normal_(self.cde_field[-1].weight, std=1e-6)
        nn.init.zeros_(self.cde_field[-1].bias)


        self.cde_alpha_raw = nn.Parameter(torch.tensor(0.0))

        # Single-step CDE correction (constant dX/dt from linear ramps)
        self._cde_ready = True
        self._alpha_scale = 1.0

    def _get_dynamics_matrix(self):
        """
        Construct the latent dynamics matrix: skew-symmetric (rotation) +
        non-positive diagonal (damping).
        """
        A_trans_skew = self.A_trans_params - self.A_trans_params.transpose(-1, -2)
        A_rot_skew = self.A_rot_params - self.A_rot_params.transpose(-1, -2)

        A_trans_diag = -F.softplus(self.A_trans_diag_raw)
        A_rot_diag = -F.softplus(self.A_rot_diag_raw)

        A_block = torch.zeros((self.latent_dyn_dim, self.latent_dyn_dim),
                              device=self.A_trans_params.device)
        A_block[:self.trans_dim, :self.trans_dim] = A_trans_skew
        A_block[self.trans_dim:, self.trans_dim:] = A_rot_skew

        idx_trans = torch.arange(self.trans_dim, device=A_block.device)
        A_block[idx_trans, idx_trans] += A_trans_diag

        idx_rot = torch.arange(self.trans_dim, self.latent_dyn_dim, device=A_block.device)
        A_block[idx_rot, idx_rot] += A_rot_diag

        return A_block

    def encode(self, v_in, x_prev=None):
        '''
        Map physical state to latent space. Position mode: v_in is 7-dim current position, x_prev is
        7-dim previous Eliminates finite-difference noise amplification (Path 3).
        '''
        if self.input_mode == 'position':
            if x_prev is None:
                x_prev = v_in
            inp = torch.cat([v_in, x_prev], dim=-1)
            return self.encoder(inp)
        return self.encoder(v_in)

    def decode(self, z):
        '''
        Map the latent state back to the physical velocity dimensions. Uses shared trunk + dual
        heads (motion & size).
        '''
        h = self.shared_trunk(z)
        motion = self.motion_head(h)
        size = self.size_head(h)
        return torch.cat([motion[..., :3], size, motion[..., 3:]], dim=-1)

    def decode_motion(self, z):
        """Return only motion components: [vx, vy, vz, vyaw]"""
        return self.motion_head(self.shared_trunk(z))

    def decode_adapted(self, z, context, return_aux=False):
        """Decode motion with the bounded P7 residual adapter.

        ``context`` is causal and has the layout
        ``[v_base(7), v_cv(7), residual_xyz(3), depth_norm, gap_norm,
        confidence, legitimacy]``.  Only motion and yaw are adapted; box size
        dynamics remain exactly those of the base ODE model.
        """
        base = self.decode(z)
        if context is None:
            if return_aux:
                zeros = torch.zeros(
                    (*base.shape[:-1], 4), device=base.device, dtype=base.dtype)
                gates = torch.zeros(
                    (*base.shape[:-1], 1), device=base.device, dtype=base.dtype)
                return base, zeros, gates
            return base
        if context.shape[-1] != self.adapter_context_dim:
            raise ValueError(
                f'Expected adapter context dim {self.adapter_context_dim}, '
                f'got {context.shape[-1]}')
        delta = self.adapter_delta_max.to(device=base.device, dtype=base.dtype)
        delta = delta * torch.tanh(self.residual_adapter(context))
        gate = 0.5 * torch.sigmoid(self.adapter_gate(context))
        adapted_motion = base[..., [0, 1, 2, 6]] + gate * delta
        output = base.clone()
        output[..., 0:3] = adapted_motion[..., 0:3]
        output[..., 6] = adapted_motion[..., 3]
        if return_aux:
            return output, delta, gate
        return output

    def decode_size(self, z):
        """Return only size components: [vl, vw, vh]"""
        return self.size_head(self.shared_trunk(z))

    def decode_with_uncertainty(self, z):
        """
        Decode latent state to physical velocity mean and per-dimension std.
        Returns (mu, sigma) each of shape (..., 7).
        sigma = softplus(raw) + 1e-3 ensures positivity.
        """
        h = self.shared_trunk(z)
        motion = self.motion_head(h)
        size = self.size_head(h)
        mu = torch.cat([motion[..., :3], size, motion[..., 3:]], dim=-1)

        raw_motion_unc = self.motion_unc_head(h)
        raw_size_unc = self.size_unc_head(h)
        sigma_motion = F.softplus(raw_motion_unc) + 1e-3
        sigma_size = F.softplus(raw_size_unc) + 1e-3
        sigma = torch.cat([sigma_motion[..., :3], sigma_size, sigma_motion[..., 3:]], dim=-1)

        return mu, sigma

    def predict(self, z_prev, delta_t):
        '''
        Evolve the latent state forward in time using matrix exponential. Applies dynamic-static
        decoupling where static features use identity mapping.
        '''
        z_dyn = z_prev[..., :self.latent_dyn_dim]
        z_stat = z_prev[..., self.latent_dyn_dim:]

        A = self._get_dynamics_matrix()

        if z_dyn.dim() > 1:

            B = z_dyn.shape[0]
            if delta_t.dim() == 0:
                dt_mat = delta_t.view(1, 1, 1).expand(B, 1, 1)
                dt_nonlin = delta_t.view(1, 1).expand(B, 1)
            else:
                dt_mat = delta_t.view(B, 1, 1)
                dt_nonlin = delta_t.view(B, 1)

            A_dt = A.unsqueeze(0) * dt_mat
            exp_A_dt = torch.matrix_exp(A_dt)
            z_dyn_next = torch.bmm(exp_A_dt, z_dyn.unsqueeze(-1)).squeeze(-1)

            nonlin_correction = self.epsilon_nonlin * self.nonlin_net(z_dyn)
            z_dyn_next = z_dyn_next + nonlin_correction * dt_nonlin
        else:
            A_dt = A * delta_t
            exp_A_dt = torch.matrix_exp(A_dt)
            z_dyn_next = torch.matmul(exp_A_dt, z_dyn)

            nonlin_correction = self.epsilon_nonlin * self.nonlin_net(z_dyn)
            z_dyn_next = z_dyn_next + nonlin_correction * delta_t

        z_stat_next = z_stat

        z_next = torch.cat([z_dyn_next, z_stat_next], dim=-1)
        return z_next

    def predict_cde(self, z_prev, X_path, t_span):
        """
        CDE-enhanced prediction: exact autonomous evolution + CDE correction.

        z_next = predict(z_prev, dt) + alpha * dt * cde_field(z_ode, dX/dt)

        Uses single-step Euler correction since training control paths are
        linear ramps (constant dX/dt). For inference with kernel-smoothed
        paths, this is a first-order approximation of the full CDE integral.

        Args:
            z_prev: (48,) or (B, 48) latent state
            X_path: (T, C) single or (B, T, C) batched control path
            t_span: (2,) or (B, 2) time span [t0, t1]

        Returns:
            z_next: (48,) or (B, 48)
        """
        batched = (X_path.dim() == 3)

        if batched:
            dt = t_span[:, 1] - t_span[:, 0]
        else:
            dt = t_span[1] - t_span[0]

        alpha = torch.sigmoid(self.cde_alpha_raw) * self._alpha_scale

        # Exact autonomous prediction
        z_ode = self.predict(z_prev, dt)
        z_dyn_ode = z_ode[..., :self.latent_dyn_dim]

        if alpha < 1e-4:
            return z_ode

        # Compute dX/dt from endpoints (constant for linear ramps)
        if batched:
            dXdt = (X_path[:, -1, :] - X_path[:, 0, :]) / dt.unsqueeze(-1)
        else:
            dXdt = (X_path[-1, :] - X_path[0, :]) / dt

        # Single-step CDE correction: alpha * dt * cde_field(z, dX/dt)
        cde_input = torch.cat([z_dyn_ode, dXdt], dim=-1)
        cde_correction = alpha * dt.unsqueeze(-1) * self.cde_field(cde_input)

        z_dyn_next = z_dyn_ode + cde_correction
        z_next = torch.cat([z_dyn_next, z_ode[..., self.latent_dyn_dim:]], dim=-1)
        return z_next

    def update(self, z_minus, v_obs, depth, x_prev=None, trust_factor=None):
        '''
        Refine the predicted latent state using the depth-aware Neural Kalman gate. Velocity mode:
        v_obs is 7-dim velocity observation. Position mode (Path 3): v_obs is 7-dim current
        position, Eliminates finite-difference noise amplification. trust_factor: scalar in [0,1]
        that scales the update magnitude.
        '''
        z_dyn_minus = z_minus[..., :self.latent_dyn_dim]
        z_stat_minus = z_minus[..., self.latent_dyn_dim:]

        if self.input_mode == 'position':
            z_obs = self.encode(v_obs, x_prev)
        else:
            z_obs = self.encode(v_obs)
        z_dyn_obs = z_obs[..., :self.latent_dyn_dim]

        soft_lambda = F.softplus(self.lambda_param)
        depth_norm = depth / (F.softplus(self.depth_norm) + 1e-6)

        if getattr(self, '_ablation_no_depth_gate', False):
            W_d = torch.ones_like(depth_norm)
        else:
            W_d = torch.exp(-soft_lambda * depth_norm)

        if W_d.dim() == 0:
            W_d = W_d.view(1, 1)
        elif W_d.dim() == 1:
            W_d = W_d.unsqueeze(-1)

        residual = torch.matmul(self.H, z_dyn_minus.unsqueeze(-1)).squeeze(-1) - z_dyn_obs

        A = self._get_dynamics_matrix()

        if z_dyn_minus.dim() > 1:
            scaled_A = A.unsqueeze(0) * W_d.unsqueeze(-1) * W_d.unsqueeze(-1)
            gate_feedback = torch.bmm(scaled_A, residual.unsqueeze(-1))
        else:
            scaled_A = A * W_d * W_d
            gate_feedback = torch.matmul(scaled_A, residual.unsqueeze(-1))

        gate_feedback = torch.matmul(self.H.transpose(-1, -2), gate_feedback)
        gate_feedback = torch.matmul(self.B, gate_feedback).squeeze(-1)

        update_term = self.epsilon * self.phi_net(gate_feedback)

        if trust_factor is not None:
            if isinstance(trust_factor, float):
                trust_factor = torch.tensor(trust_factor, device=update_term.device)
            update_term = update_term * trust_factor

        z_dyn_plus = z_dyn_minus - update_term

        z_plus = torch.cat([z_dyn_plus, z_stat_minus], dim=-1)
        return z_plus


class _SimpleLinearInterp:
    """Minimal linear interpolation fallback when torchcde is unavailable."""

    def __init__(self, X, t):
        '''Initialize the object state and configure the resources required by this component.'''
        self.X = X
        self.t = t

    def derivative(self, t_query):
        '''Handle the derivative step for this module's workflow.'''
        idx = torch.searchsorted(self.t, t_query)
        idx = torch.clamp(idx, 1, len(self.t) - 1)
        t_left = self.t[idx - 1]
        t_right = self.t[idx]
        dt = t_right - t_left
        if dt < 1e-8:
            return torch.zeros(self.X.shape[1], device=self.X.device)
        return (self.X[idx] - self.X[idx - 1]) / dt


class _BatchedLinearInterp:
    """Batched linear interpolation for (B, T, D) control paths."""

    def __init__(self, X, t):
        '''Initialize the object state and configure the resources required by this component.'''
        self.X = X
        self.t = t

    def derivative(self, t_query):
        '''Handle the derivative step for this module's workflow.'''
        idx = torch.searchsorted(self.t, t_query)
        idx = torch.clamp(idx, 1, len(self.t) - 1)
        t_left = self.t[idx - 1]
        t_right = self.t[idx]
        dt = t_right - t_left
        if dt < 1e-8:
            return torch.zeros(self.X.shape[0], self.X.shape[2], device=self.X.device)
        return (self.X[:, idx, :] - self.X[:, idx - 1, :]) / dt

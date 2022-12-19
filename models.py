from torch.nn.functional import relu, interpolate
import torch
import torch.nn as nn

import torch_cubic_spline_interp


class SeriesExpansion(nn.Module):
    def __init__(self, P, N, K, num_instances, L_out, interp_mode, learning_mode, theta, domain='spatial', pi_symm=True,
                 Theta_type='harmonic'):
        super(SeriesExpansion, self).__init__()

        self.relu = nn.ReLU()
        self.P = P
        self.K = K
        self.N = N
        self.L_out = L_out
        self.num_instances = num_instances
        self.interp_mode = interp_mode
        self.learning_mode = learning_mode
        self.theta = torch.Tensor(theta).cuda()
        self.pi_symm = pi_symm
        self.Theta_type = Theta_type

        if learning_mode in ['only_temp_rep_learn_linear', 'only_temp_rep_learn_poly', 'only_temp_rep_learn_sine',
                             'only_temp_rep_cubic_spline', 'only_temp_rep_learn_casorati']:
            self.temporal_fcts = torch.autograd.Variable(torch.randn(1, K + 1, L_out).cuda(), requires_grad=True)
            self.conv_layer = nn.ModuleList([nn.Identity() for i in range(0, K + 1)])

        # C matrix with view pattern
        if Theta_type == 'harmonic':
            self.Theta_mtx = torch.zeros([1, self.P, 2 * self.N + 1], dtype=torch.cfloat).cuda()
            self.Theta_mtx_symm = torch.zeros([1, self.P, 2 * self.N + 1], dtype=torch.cfloat).cuda()
            for n in range(2 * self.N + 1):
                self.Theta_mtx[0, :, n] = torch.exp(1j * (n - self.N) * self.theta)
                self.Theta_mtx_symm[0, :, n] = torch.exp(1j * (n - self.N) * (self.theta - torch.pi))
            print(r'$\kappa(Theta)$:', torch.linalg.cond(self.Theta_mtx), r'$\kappa(Theta)$ symm:',
                  torch.linalg.cond(torch.cat((self.Theta_mtx, self.Theta_mtx_symm), dim=2)))

        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.L_out).cuda()
        self.P_t = torch.linspace(0, 1, self.P).cuda()

        # training domain
        self.domain = domain

    def forward(self, g_t):
        g_t = g_t.type(torch.cfloat)

        if self.pi_symm:
            L1 = torch.zeros([2 * self.P, (2 * self.N + 1) * (self.K + 1)], dtype=torch.cfloat).cuda()
        else:
            L1 = torch.zeros([self.P, (2 * self.N + 1) * (self.K + 1)], dtype=torch.cfloat).cuda()

        b = torch.zeros([g_t.shape[-2], 2 * self.N + 1, self.P], dtype=torch.cfloat).cuda()
        f_pol_est_s = torch.zeros([g_t.shape[-2], self.P, self.num_instances], dtype=torch.cfloat).cuda()

        psi_mtx = torch.zeros([1, self.K + 1, self.P], dtype=torch.cfloat).cuda()
        if self.learning_mode == 'only_temp_rep_learn_linear':
            for k in range(self.K + 1):
                psi_mtx[0, k, :] = self.conv_layer[k](
                    interpolate(self.temporal_fcts[:, k, :].view(1, 1, self.L_out), size=self.P, mode=self.interp_mode))
        elif self.learning_mode == 'only_temp_rep_cubic_spline':
            for k in range(self.K + 1):
                psi_mtx[0, k, :] = self.conv_layer[k](
                    torch_cubic_spline_interp.interp(self.D_t, self.temporal_fcts[:, k, :].view(self.L_out),
                                                     self.P_t).view(1, 1, self.P))

        for k in range(self.K + 1):
            for n in range(b.shape[1]):
                L1[:self.P, n * (self.K + 1) + k] = self.Theta_mtx[0, :, n] * psi_mtx[0, k, :]
                if self.pi_symm:
                    L1[self.P:, n * (self.K + 1) + k] = self.Theta_mtx_symm[0, :, n] * psi_mtx[0, k, :]

        eta = (torch.linalg.pinv(L1) @ torch.transpose(g_t, 2, 3))
        g_est = L1 @ eta

        with torch.no_grad():
            # Compute b_n
            for n in range(b.shape[1]):
                for k in range(self.K + 1):
                    b[:, n, :] += (torch.outer(eta[0, 0, n * (self.K + 1) + k, :], psi_mtx[0, k, :])).view(
                        g_t.shape[-2], self.P)

            # Compute F_pol_est
            for instance in range(self.num_instances):
                for n in range(0, b.shape[1]):
                    f_pol_est_s[:, :, instance] += torch.outer(b[:, n, instance * self.P // self.num_instances],
                                                               self.Theta_mtx[0, :, n]).view(g_t.shape[-2], self.P)

        return eta, L1, g_est.transpose(2, 3), b, f_pol_est_s, psi_mtx

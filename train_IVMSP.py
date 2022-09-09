import copy
import numpy as np
import torch


def train_theta_eta_simult(model, optimizer_latent, criterion_g, criterion_psi,
                           criterion_f, theta_exp, f_pol_s, g, learning_mode, training_mode,
                           sgd_mode, P, K, L_out, N, num_epoch, spatial_dim,
                           ortho_weight=1e-6, verbose=True):
    # Initialize loss vectors
    min_f_pol_s_epoch = 1e8
    spatial_idx = np.arange(spatial_dim)
    loss_epoch = []
    loss_temporal_epoch = []
    loss_g_epoch = []
    loss_Theta_epoch = []
    loss_f_pol_s_epoch = [min_f_pol_s_epoch]

    # Convert inputs to torch tensor (cuda)
    theta_gpu = torch.Tensor(theta_exp).cuda()
    f_pol_s_gpu = torch.Tensor(f_pol_s).cuda()

    # Single sample per time instant setting
    if training_mode == 'single_view':
        g_gt = torch.Tensor(g).view(1, 1, g.shape[0], g.shape[1]).cuda()

    I_temporal = torch.eye(K + 1).cuda()

    if learning_mode in ['only_temp_rep_learn_linear', 'only_temp_rep_cubic_spline']:
        t_gpu_latent = [torch.autograd.Variable(torch.randn(1, 1, P), requires_grad=True) for i in range(K + 1)]

    print('K:', K, ' N:', N, ' L_out:', L_out, ' P:', P, 'spatial_dim', spatial_dim, ' Training mode:',
          training_mode, ' Learning mode:', learning_mode, 'sgd_mode:', sgd_mode)

    for epoch in range(num_epoch):
        loss_sum = 0
        loss_temporal_sum = 0
        loss_Theta_sum = 0
        loss_g_sum = 0
        loss_f_pol_s_sum = 0
        np.random.shuffle(spatial_idx)

        if training_mode == 'single_view':

            # Call forward model to generate eta, Theta, g_est, and f_pol_est
            if sgd_mode in ['gd']:
                eta_est, L1_est, g_est, b_est, f_pol_s_est, psi_mtx = model(g_gt)
                # Compute measurement estimation error
                loss_g = criterion_g(g_est.real, g_gt) + criterion_g(g_est.imag, torch.zeros(g_gt.shape).cuda())

            # Compute recon error but not gradients.
            with torch.no_grad():
                loss_f_pol_s = criterion_f(f_pol_s_est.real, f_pol_s_gpu)

            # Enforce orthonormality for temporal fcts. for non/linear approach
            if learning_mode in ['only_temp_rep_learn_linear', 'only_temp_rep_cubic_spline']:
                T = torch.zeros(L_out, K + 1).cuda()
                for k in range(K + 1):
                    # Enforce Z ortho
                    T[:, k] = model.temporal_fcts[:, k, :].view(1, 1, L_out)

            loss_temporal = ortho_weight * criterion_psi(T.transpose(0, 1) @ T, I_temporal)

            # Compute overall loss
            loss = loss_g + loss_temporal

            # Backprop
            optimizer_latent.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_latent.step()

            # Log computed losses
            loss_sum += loss.data.cpu().numpy()
            loss_temporal_sum += loss_temporal.data.cpu().numpy()
            loss_g_sum += loss_g.data.cpu().numpy()
            loss_f_pol_s_sum += loss_f_pol_s.data.cpu().numpy()

            loss_epoch.append(loss_sum / spatial_dim)
            loss_temporal_epoch.append(loss_temporal_sum / spatial_dim)
            loss_g_epoch.append(loss_g_sum / spatial_dim)
            loss_f_pol_s_epoch.append(loss_f_pol_s_sum / spatial_dim)
            loss_Theta_epoch.append(loss_Theta_sum / spatial_dim)

            if loss_f_pol_s_epoch[-1] < min_f_pol_s_epoch:
                min_f_pol_s_epoch = loss_f_pol_s_epoch[-1] * 1
                best_model = copy.deepcopy(model)
                best_psi_mtx = psi_mtx * 1

            if epoch % 2 == 0 and verbose:
                print('Epoch: ', epoch, 'total loss: %.3e /' % loss_epoch[-1],
                      'temp. ortho. loss: %.3e /' % loss_temporal_epoch[-1], 'g loss: %.3e /' % loss_g_epoch[-1],
                      'f_pol loss: %.3e /' % loss_f_pol_s_epoch[-1], 'cond(L1): %.3e /' % torch.linalg.cond(L1_est),
                      'Theta ortho. loss: %.3e /' % loss_Theta_epoch[-1])

    return best_model, loss_epoch, loss_temporal_epoch, loss_g_epoch, loss_f_pol_s_epoch, loss_Theta_epoch, g_gt, t_gpu_latent, theta_gpu, L1_est, best_psi_mtx


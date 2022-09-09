import numpy as np
from skimage.transform import radon, iradon
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import relu, interpolate
import torch
import torch_cubic_spline_interp


def DecimalToBinary(n):
    return bin(n).replace("0b", "")


def bit_reversal(x, N):
    num_digit = 0
    while N // 2:
        N = N // 2
        num_digit += 1
    x = list(DecimalToBinary(x))
    while len(x) < num_digit:
        x = [0] + x
    x.reverse()
    return int("".join(str(n) for n in x), 2)


def obtain_beta_coeffs(Theta, g):
    beta_est = np.squeeze(np.linalg.pinv(Theta) @ g[:, :, np.newaxis]).T
    return beta_est


def obtain_bn(beta_est, W, t, N, P, K, Psi, temporal_mode='mtx'):
    bn_est = np.zeros([2 * N + 1, W, P], dtype=complex)  # Obtain estimates of b_n[t]'s.
    for n in range(2 * N + 1):
        for k in range(K + 1):
            if temporal_mode == 'fct':
                bn_est[n, :, :] += np.outer(beta_est[(n * (K + 1)) + k, :], Psi(t[:], k))
            elif temporal_mode == 'mtx':
                bn_est[n, :, :] += np.outer(beta_est[(n * (K + 1)) + k, :], Psi[:, k])
    return bn_est


def form_g(bn_est, W, theta, N, P, K):
    g_est = np.zeros([W, P], dtype=complex)
    for n in range(0, 2 * N + 1):
        g_est += bn_est[n, :, :] * np.exp(1j * (n - N) * theta[:])
    return g_est


def form_F_pol(eta_est, G, N, P, K, t, Psi, num_instances, theta, temporal_mode='mtx'):
    hn_est = obtain_bn(eta_est, G.shape[0], t, N, P, K, Psi, temporal_mode)
    F_pol_est = np.zeros([G.shape[0], G.shape[1], num_instances], dtype=complex)
    for instance in range(num_instances):
        for n in range(0, 2 * N + 1):
            F_pol_est[:, :, instance] += np.outer(hn_est[n, :, instance * P // num_instances],
                                                  np.exp(1j * (n - N) * theta[:]))
    return F_pol_est, hn_est


def obtain_projections(f, theta, P):
    g = np.zeros([f.shape[0], P])
    for p in range(P):
        g[:, p] = radon(f[:, :, p], theta=[360 * theta[p] / (2 * np.pi)])[:, 0]
    print('g norm: ', np.linalg.norm(g))
    return g


def obtain_all_projections(f, theta, P):
    g = np.zeros([f.shape[0], len(theta), P])
    for p in range(P):
        g[:, :, p] = radon(f[:, :, p], theta=360 * theta / (2 * np.pi))
    return g


def obtain_plain_reconstruction(g, theta):
    return iradon(g, theta=360 * theta / (2 * np.pi))


def form_Theta_w_funct_Psi(theta, polynomial, t, P, K, N):
    Theta = np.zeros([P, (2 * N + 1) * (K + 1)], dtype=complex)
    for k in range(K + 1):
        for n in range(2 * N + 1):
            Theta[:, n * (K + 1) + k] = np.exp(1j * (n - N) * theta[:]) * polynomial(t[:], k)
    return Theta


def form_Theta_w_mtx_Psi(theta, Psi, P, K, N):
    Theta = np.zeros([P, (2 * N + 1) * (K + 1)], dtype=complex)
    for k in range(K + 1):
        for n in range(2 * N + 1):
            Theta[:, n * (K + 1) + k] = np.exp(1j * (n - N) * theta[:]) * Psi[:, k]
    return Theta


def compute_psnr(x_gt, x_rec):
    mse = np.mean((x_rec - x_gt) ** 2)
    psnr = 20 * np.log10((np.max(x_gt) - np.min(x_gt)) / np.sqrt(mse))
    return psnr


def compute_mae(x_gt, x_rec):
    return np.mean(np.abs(x_rec - x_gt))


def compute_ssim(x_gt, x_rec):
    return ssim(x_gt, x_rec, data_range=x_rec.max() - x_rec.min())


def display_f(f, image, rate, P):
    Psqrt = np.int(np.ceil(np.sqrt(P / rate)))
    plt.figure(figsize=(15, 15))
    for p in range(P):
        if p % rate == 0:
            plt.subplot(Psqrt, Psqrt, p // rate + 1)
            plt.imshow(f[:, :, p], cmap='gray')
            plt.clim(0, np.max(image))
    plt.show()


def generate_theta(P, ang_range=2 * np.pi):
    theta_linear = np.linspace(0, ang_range, P)  # uniform sampling of angle.
    theta_random = np.random.uniform(0, ang_range, size=[P])  # random sampling of angle.
    theta_bit_reversal = []  # bit-reversal scheme for angle picking.
    theta_golden_angle = []
    for p in range(P):
        theta_golden_angle.append((p * (111.25 / 360) * 2 * np.pi) % (2 * np.pi))
        theta_bit_reversal.append((ang_range / P) * bit_reversal(p, P))
    theta_bit_reversal = np.array(theta_bit_reversal)
    theta_golden_angle = np.array(theta_golden_angle)
    return theta_linear, theta_random, theta_bit_reversal, theta_golden_angle


def form_F_pol_from_hn(hn_est, G, N, P, num_instances, theta):
    F_pol_est = np.zeros([G.shape[0], G.shape[1], num_instances], dtype=complex)
    for instance in range(num_instances):
        for p in range(P):
            for n in range(0, hn_est.shape[0]):
                F_pol_est[:, p, instance] += hn_est[n, :, instance * P // num_instances] * np.exp(
                    1j * (n - N) * theta[p])
    return F_pol_est, hn_est


def form_F_pol_temporal_cuda(eta_est, G, N, P, K, L_out, t, temporal_fct, num_instances, theta, interp_mode,
                             conv_fct=None, model=None):
    hn_est = obtain_bn_temporal_cuda(eta_est, G.shape[0], P, K, L_out, temporal_fct, interp_mode, conv_fct,
                                     model)
    F_pol_est = torch.zeros([G.shape[0], G.shape[1], num_instances], dtype=torch.cfloat).cuda()
    for instance in range(num_instances):
        if instance % 60 == 0:
            print(instance)
        for n in range(0, hn_est.shape[0]):
            F_pol_est[:, :, instance] += torch.outer(hn_est[n, :, instance * P // num_instances],
                                                     torch.exp(1j * (n - N) * theta[:]))
    return F_pol_est, hn_est


def obtain_bn_temporal_cuda(beta_est, W, P, K, L_out, temporal_fct, interp_mode, conv_fct=None, model=None):
    print(beta_est.shape)
    bn_est = torch.zeros([beta_est.shape[0] // (K + 1), W, P],
                         dtype=torch.cfloat).cuda()  # Obtain estimates of b_n[t]'s.
    for n in range(beta_est.shape[0] // (K + 1)):
        for k in range(K + 1):
            if interp_mode == 'only_temp_rep_learn_linear':
                f_t = conv_fct[k](interpolate(temporal_fct[:, k, :].view(1, 1, L_out), size=P, mode='linear'))
            elif interp_mode == 'only_temp_rep_cubic_spline':
                f_t = conv_fct[k](
                    torch_cubic_spline_interp.interp(model.D_t, temporal_fct[:, k, :].view(L_out), model.P_t).view(1, 1,
                                                                                                                   P))
            bn_est[n, :, :] += torch.outer(beta_est[(n * (K + 1)) + k, :], f_t[0, 0, :])
    return bn_est


def plot_FBP_results(f, f_true_recon, f_pol_est_static_FBP, f_pol_est_FBP, P, num_instances, rate,
                     exp_name, val_cbar='max'):
    f_gt = np.zeros([f.shape[0], f.shape[1], num_instances])
    for instant in range(num_instances):
        f_gt[:, :, instant] = f[:, :, instant * P // num_instances]
    psnr_dynamic = compute_psnr(f_gt, f_pol_est_FBP)
    psnr_dynamic_recon = compute_psnr(f_true_recon, f_pol_est_FBP)

    plt.figure(figsize=(3 * num_instances / rate, 3 * 2))
    for instant in range(num_instances // rate):
        plt.subplot(1, num_instances // rate, instant + 1)
        plt.title('f %d' % instant)
        plt.imshow(f[:, :, instant * P * rate // num_instances], cmap='gray')
        plt.clim(0, np.max(f))
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.title('F_{pol} estimation static FBP %s' % exp_name)
    plt.imshow(f_pol_est_static_FBP, cmap='gray')
    plt.clim(0, np.max(f))
    plt.show()

    plt.figure(figsize=(5 * num_instances / rate, 5 * 4))
    plt.suptitle('PSNR recon=%.3f dB' % psnr_dynamic_recon)
    for instant in range(num_instances // rate):
        max_val_cbar = f.max() if val_cbar == 'max' else f[:, :, instant * P * rate // num_instances].max()
        plt.subplot(5, num_instances // rate, instant + 1)
        plt.title('f %d' % instant)
        plt.imshow(f[:, :, instant * P * rate // num_instances], cmap='gray')
        plt.clim(0, max_val_cbar)
        plt.colorbar()
        plt.subplot(5, num_instances // rate, instant + 1 + num_instances // rate)
        plt.title('f_est %d' % (instant))
        plt.imshow(f_pol_est_FBP[:, :, instant * P * rate // num_instances], cmap='gray')
        plt.clim(0, max_val_cbar)
        plt.colorbar()
        plt.subplot(5, num_instances // rate, instant + 1 + 2 * num_instances // rate)
        plt.title('f_true_recon %d' % (instant))
        plt.imshow(f_true_recon[:, :, instant * P * rate // num_instances], cmap='gray')
        plt.clim(0, max_val_cbar)
        plt.colorbar()
        plt.subplot(5, num_instances // rate, instant + 1 + 3 * num_instances // rate)
        plt.title('|f_true_recon - f_est| %d' % (instant))
        plt.imshow(np.abs(f_pol_est_FBP[:, :, instant * P * rate // num_instances] - f_true_recon[:, :,
                                                                                     instant * P * rate // num_instances]),
                   cmap='gray')
        plt.clim(0, max_val_cbar)
        plt.colorbar()
        plt.subplot(5, num_instances // rate, instant + 1 + 4 * num_instances // rate)
        plt.title('|f_true_recon - f_est| %d' % (instant))
        plt.imshow(np.abs(f_pol_est_FBP[:, :, instant * P * rate // num_instances] - f_true_recon[:, :,
                                                                                     instant * P * rate // num_instances]),
                   cmap='gray')
        plt.clim(0, max_val_cbar / 4)
        plt.colorbar()
        plt.tight_layout()
    plt.show()

    return psnr_dynamic, psnr_dynamic_recon


def compute_psnr(x_gt, x_rec):
    mse = np.mean((x_rec - x_gt) ** 2)
    return 20 * np.log10((np.max(x_gt) - np.min(x_gt)) / np.sqrt(mse))

import numpy as np

from stochastic_spiking.visualization_tools.theoretical_tools import isi, signal_definition, hit, mutau_step, \
    mutau_step_infinitesimal


def define_h(tz=0., tt=0., tau_m=0.01, tau=.1, A_s=1):
    def h(z, t):
        return A_s * (1 + np.cos(z - tz)) * (
                np.heaviside(t - tt, .5) * (1 - np.exp(-(t - tt) / tau_m))
                - np.heaviside(t - tt - tau, .5) * (1 - np.exp(-(t - tt - tau) / tau_m))
        )

    return h


def define_h_cev(tz=0., tt=0., tau_m=0.01, tau=.1, A_s=1):
    def h(z, t):
        return A_s * (1 + np.cos(z - tz)) * (
                np.heaviside(t - tt, .5)
                - np.heaviside(t - tt - tau, .5)
        )

    return h


def define_dhdz(theta, tz=0., tt=0., tau_m=0.01, tau=.1, A_s=1.):
    def dh_dz(z, t):
        if theta == 'position':
            d = A_s * np.sin(z - tz) * (
                    np.heaviside(t - tt, .5) * (1 - np.exp(-(t - tt) / tau_m))
                    - np.heaviside(t - tt - tau, .5) * (1 - np.exp(-(t - tt - tau) / tau_m))
            )

        elif theta == 'time':
            d = A_s / tau_m * (1 + np.cos(z - tz)) * (
                    - np.heaviside(t - tt, .5) * np.exp(-(t - tt) / tau_m)
                    + np.heaviside(t - tt - tau, .5) * np.exp(-(t - tt - tau) / tau_m)
            )

        elif theta == 'amplitude':
            d = (1 + np.cos(z - tz)) * (
                    np.heaviside(t - tt, .5) * (1 - np.exp(-(t - tt) / tau_m))
                    - np.heaviside(t - tt - tau, .5) * (1 - np.exp(-(t - tt - tau) / tau_m))
            )

        else:
            raise NotImplementedError

        return d

    return dh_dz


def define_dhdz_cev(theta, tz=0., tt=0., tau_m=0.01, tau=.1, A_s=1.):
    def dh_dz(z, t):
        if theta == 'position':
            d = A_s * np.sin(z - tz) * (
                    np.heaviside(t - tt, .5)
                    - np.heaviside(t - tt - tau, .5)
            )

        elif theta == 'time':
            raise NotImplementedError

        elif theta == 'amplitude':
            d = (1 + np.cos(z - tz)) * (
                    np.heaviside(t - tt, .5)
                    - np.heaviside(t - tt - tau, .5)
            )

        else:
            raise NotImplementedError

        return d

    return dh_dz


def define_eta_taro(g_M, beta, g, h, dh_dz):
    def eta(z, t):
        s = g(h(z, t)) / g_M
        eta = beta ** 3 * (dh_dz(z, t) * (1 - s)) ** 2 * (1 - 3 * s)
        # eta = beta ** 3 * (dh_dz(z, t)) ** 2
        return eta

    return eta


def define_integrand_weight(nu_eta, nu_0, tau_m=0.01, gamma=1):
    def integrand_weight(t, z, t_, zeta):
        epsilon = np.exp(-(t - t_) / tau_m) * np.heaviside(t - t_, .5)
        dj = epsilon * nu_eta(z, t) * nu_0(z - zeta, t_) * gamma
        return dj

    return integrand_weight


def parallel_append_weights(t_int, z_int, t__int, zeta, parallel_weights, vt, vzeta, vt_, integrand_weight):
    r = integrand_weight(t_int, z_int, t__int, zeta)
    nonnans = ~np.isnan(r)
    r = r[nonnans]
    r = r.mean() * vt * vzeta * vt_
    print(r)
    parallel_weights.append(r)
    return r


def define_integrand_J_0(dh_dz, gp2_g, gamma):
    def integrand_J_0(t, z):
        j_0 = (dh_dz(z, t)) ** 2 * gp2_g(z, t) * gamma

        nonnans = ~np.isnan(j_0)
        j_0 = j_0[nonnans]
        return j_0

    return integrand_J_0


def define_p_isi_cev(v0, sigma, rho, A_s, z_0, tau_s, t_0):
    def p_isi_cev(z, t):
        amplitude_steps = A_s * (1 + np.cos(z - z_0))
        amplitudes = [amplitude_steps, -amplitude_steps]
        mu_, tau_ = mutau_step(period=tau_s, phase=t_0, amplitudes=amplitudes)

        pisi = isi(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_, tau_=tau_, library_derivatives='tensorflow')
        return pisi

    return p_isi_cev


def define_p_isi_cev_derivatives(v0, sigma, rho, A_s, z_0, tau_s, t_0, delta):
    def p_isi_cev_derivative(z, t):
        amplitude_steps = A_s * (1 + np.cos(z - z_0))
        amplitudes = [amplitude_steps, -amplitude_steps]
        mu_, tau_ = mutau_step(period=tau_s, phase=t_0, amplitudes=amplitudes)
        mu_d, tau_d = mutau_step_infinitesimal(period=tau_s, phase=t_0, amplitudes=amplitudes, delta=delta)

        pisi = isi(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_, tau_=tau_, library_derivatives='tensorflow')
        pisi_d = isi(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_d, tau_=tau_d, library_derivatives='tensorflow')

        d = (pisi_d - pisi) / delta
        return d

    def p_isi_cev_derivative2(z, t):
        amplitude_steps = A_s * (1 + np.cos(z - z_0))
        amplitudes = [amplitude_steps, -amplitude_steps]
        mu_, tau_ = mutau_step(period=tau_s, phase=t_0, amplitudes=amplitudes)
        mu_d, tau_d = mutau_step_infinitesimal(period=tau_s, phase=t_0, amplitudes=amplitudes, delta=delta)
        mu_2d, tau_2d = mutau_step_infinitesimal(period=tau_s, phase=t_0, amplitudes=amplitudes, delta=2 * delta)

        pisi = isi(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_, tau_=tau_, library_derivatives='tensorflow')
        pisi_d = isi(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_d, tau_=tau_d, library_derivatives='tensorflow')
        pisi_2d = isi(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_2d, tau_=tau_2d, library_derivatives='tensorflow')

        d2 = (pisi_2d - 2 * pisi_d + pisi) / delta ** 2
        return d2

    return p_isi_cev_derivative, p_isi_cev_derivative2


def define_p_hit_cev(v0, sigma, rho, A_s, z_0, tau_s, t_0):
    def p_hit_cev(z, t):
        amplitude_steps = A_s * (1 + np.cos(z - z_0))
        mu_, tau_, _, _ = signal_definition('one_step', q=tau_s, amplitude_steps=amplitude_steps, phase=t_0)

        phit = hit(t, v0=v0, sigma=sigma, beta=rho, mu_=mu_, tau_=tau_, library_derivatives='tensorflow')
        return phit

    return p_hit_cev


def define_task(noise_type, z_0, t_0, tau_m, tau_s, A_s, theta, g_M, beta, u_c, v0, sigma, rho):
    if noise_type == 'taro':
        h = define_h(tz=z_0, tt=t_0, tau_m=tau_m, tau=tau_s, A_s=A_s)
        dh_dz = define_dhdz(theta, tz=z_0, tt=t_0, tau_m=tau_m, tau=tau_s, A_s=A_s)

        sigmoid = lambda x: (1 + np.exp(-x)) ** (-1)
        g = lambda x: g_M * sigmoid(beta * (x - u_c))
        nu_0 = lambda z, t: g(h(z, t))
        # g_low = lambda x: g_M * np.exp(beta * (x - u_c))
        eta = define_eta_taro(g_M, beta, g, h, dh_dz)
        # gp = lambda x: beta * g(x) * (1 - g(x) / g_M)
        gp2_g = lambda z, t: (beta * (1 - g(h(z, t)) / g_M)) ** 2 * g(h(z, t))
        nu_eta = lambda z, t: eta(z, t) * nu_0(z, t)
    elif noise_type == 'cev':
        import tensorflow as tf

        tf.keras.backend.set_floatx('float32')
        h = define_h_cev(tz=z_0, tt=t_0, tau_m=tau_m, tau=tau_s, A_s=A_s)
        dh_dz = define_dhdz_cev(theta, tz=z_0, tt=t_0, tau_m=tau_m, tau=tau_s, A_s=A_s)
        epsilon = 1e-8
        p_isi = define_p_isi_cev(v0, sigma, rho, A_s, z_0, tau_s, t_0)
        p_hit = define_p_hit_cev(v0, sigma, rho, A_s, z_0, tau_s, t_0)
        g = p_isi
        nu_0 = lambda z, t: (p_isi(z, t) / (p_hit(z, t) + epsilon)).numpy()

        delta = .001
        dp_isi, d2p_isi = define_p_isi_cev_derivatives(v0, sigma, rho, A_s, z_0, tau_s, t_0, delta)
        gp2_g = lambda z, t: (dp_isi(z, t) ** 2 / (p_isi(z, t) + epsilon)).numpy()
        nu_eta = lambda z, t: ((dh_dz(z, t) * dp_isi(z, t)) ** 2 / (p_isi(z, t) + epsilon) * (
                2 * d2p_isi(z, t) / (dp_isi(z, t) + epsilon) + dp_isi(z, t) / (p_isi(z, t) + epsilon))).numpy()

        # dp = dp_isi(z_int, t_int).numpy()
        # print('p_isi:     ', p_isi(z_int, t_int).numpy().mean())
        # print('p_hit:     ', p_hit(z_int, t_int).numpy().mean())
        # print('nu_0:      ', nu_0(z_int, t_int).mean())
        # print('dp_isi:   ', dp.mean())
        # print('dp_isi:       ', (np.count_nonzero(~np.isnan(dp))))
        # print('dp_isi:       ', (np.count_nonzero(np.isnan(dp))))
        # print('d2p_isi:  ', d2p_isi(z_int, t_int).numpy().mean())
        # print('gp2_g:    ', gp2_g(z_int, t_int).mean())
        # print('eta:      ', nu_eta(z_int, t_int).mean())
        # print('dh_dz:    ', dh_dz(z_int, t_int).mean())
        # print('dh*dp:    ', ((dh_dz(z_int, t_int) * dp_isi(z_int, t_int)) ** 2).numpy().mean())
        # print('dh*dp:    ', ((dh_dz(z_int, t_int) * dp_isi(z_int, t_int))).numpy().max())
        # print('gpp/gp:   ', (d2p_isi(z_int, t_int) / (dp_isi(z_int, t_int) + epsilon)).numpy().mean())
        # print('gp/g:     ', (dp_isi(z_int, t_int) / (p_isi(z_int, t_int) + epsilon)).numpy().mean())
        # j = integrand_weight(t_int, z_int, t__int, .1)
        # print('J1:       ', (j).mean())
        # print('nnans:    ', (np.count_nonzero(np.isnan(j))))


    else:
        raise NotImplementedError

    return gp2_g, nu_eta, dh_dz, nu_0

import tensorflow as tf

import sympy as sp
import numpy as np

from stochastic_spiking.neural_models.pseudoderivatives import SpikeFunctionDeltaDirac


def ld_operations(library_derivatives):
    if library_derivatives in ['tensorflow', 'tf']:
        ld = tf
        ld.abs = tf.abs
        ld.gamma = lambda x: tf.exp(tf.math.lgamma(x))
        ld.compile = lambda f, x: tf.keras.models.Model(x, f)

        def evaluate(f, x0):
            # I'm going to assume t is the first element of the list
            if len(x0) == 1:
                standardized = np.array(x0)[None]
            else:
                time_steps = x0[0].shape[0]
                standardized = [x0[0][..., None]] + [np.repeat(x, time_steps)[..., None] for x in x0[1:]]
                # return f(standardized).numpy()[:, 0].tolist()
            output = f(standardized)[:, 0]
            output = output.eval(session=tf.compat.v1.Session())
            return output

        ld.evaluate = lambda f, x, x0: evaluate(f, x0)
        ld.for_evaluate = ld.evaluate
        ld.heaviside = lambda x, y: SpikeFunctionDeltaDirac(x, 1.)
        ld.pow = tf.pow
        ld.sum = tf.reduce_sum
        ld.rand = tf.cast(tf.random.uniform([]), tf.keras.backend.floatx())
        ld.log = tf.math.log

    elif library_derivatives in ['sympy', 'sp']:
        ld = sp
        ld.abs = sp.Abs
        ld.gamma = sp.gamma
        ld.evaluate = lambda f, x, x0: f.subs({x_i: x0_i for x_i, x0_i in zip(x, x0)})
        ld.for_evaluate = lambda f, x, x0: [ld.evaluate(f, x, [i, *x0[1:]]) for i in x0[0]]
        ld.heaviside = sp.Heaviside
        ld.compile = lambda x, y: x
        ld.pow = lambda x, y: x ** y
        ld.sum = sum
        ld.rand = np.random.uniform()

    else:
        raise NotImplementedError

    return ld


def fisher_info(pd, score, param):
    f = pd * score ** 2
    return f


def mutau_step(period, phase, amplitudes=[1, -1], library_derivatives='tensorflow'):
    ld = ld_operations(library_derivatives)

    def A_i_step_linear(t, i, beta, acc_amplitude):
        current = (t - period * i - phase)
        return current

    def A_i_step(t, i, beta, acc_amplitude):
        e = ld.exp(2 * beta * acc_amplitude * (t - period * i - phase))
        current = (e - 1) / (2 * beta * acc_amplitude)
        current = tf.cast(current, dtype=tf.keras.backend.floatx())
        return current

    def mu(t):
        output = 0
        for i, amplitude in enumerate(amplitudes):
            output += amplitude * ld.heaviside(t - period * i - phase, .5) * (t - period * i - phase)
        return output

    def tau(t, beta):
        output = t  # t
        previous = t - phase
        time_passed = phase
        for i, amplitude in enumerate(amplitudes[:-1]):
            acc_amplitude = sum(amplitudes[:i + 1])
            current = A_i_step(t, i, beta, acc_amplitude)
            output += (current - previous) * ld.heaviside(t - period * i - phase, .5)

            time_passed += period
            current_evaluated = A_i_step(time_passed, i, beta, acc_amplitude)
            previous = current - current_evaluated

        # linear final part
        # acc_amplitude = sum(amplitudes[:-1])
        current = A_i_step_linear(t, i + 1, beta, 0)
        output += (current - previous) * ld.heaviside(t - period * i - phase, .5)

        return output

    return mu, tau


def mutau_step_infinitesimal(period, phase, amplitudes=[1, -1], library_derivatives='tensorflow', delta=.01):
    ld = ld_operations(library_derivatives)

    def A_i_step_linear(t, i, beta, acc_amplitude):
        current = (t - period * i - phase)
        current = tf.cast(current, dtype=tf.keras.backend.floatx())

        return current

    def A_i_step_squared(t, i, beta, acc_amplitude):
        current = (t - period * i - phase)
        current = tf.cast(current, dtype=tf.keras.backend.floatx())

        return current ** 2

    def A_i_step(t, i, beta, acc_amplitude):
        e = ld.exp(2 * beta * acc_amplitude * (t - period * i - phase))
        current = (e - 1) / (2 * beta * acc_amplitude)
        current = tf.cast(current, dtype=tf.keras.backend.floatx())

        return current

    def A_i_step_delta(t, i, beta, acc_amplitude):
        e = ld.exp(2 * beta * acc_amplitude * (t - period * i - phase)) * (
                2 * beta * acc_amplitude * (t - period * i - phase) - 1)
        current = (e + 1) / (2 * beta * acc_amplitude) ** 2
        current = tf.cast(current, dtype=tf.keras.backend.floatx())

        return current

    def mu(t):
        output = 0
        for i, amplitude in enumerate(amplitudes):
            output += amplitude * ld.heaviside(t - period * i - phase, .5) * (t - period * i - phase)
        return output + delta * t

    def tau(t, beta):
        output = t + 2 * beta * delta * t ** 2 / 2
        previous = t + 2 * beta * delta * t ** 2 / 2 - phase
        time_passed = phase
        for i, amplitude in enumerate(amplitudes[:-1]):
            acc_amplitude = sum(amplitudes[:i + 1])
            current = A_i_step(t, i, beta, acc_amplitude) \
                      + 2 * beta * delta * A_i_step_delta(t, i, beta, acc_amplitude)
            output += (current - previous) * ld.heaviside(t - period * i - phase, .5)

            time_passed += period
            current_evaluated = A_i_step(time_passed, i, beta, acc_amplitude) \
                                + 2 * beta * delta * A_i_step_delta(time_passed, i, beta, acc_amplitude)
            previous = current - current_evaluated

        # linear final part
        # it would have been cooler to write the code with an if statement that chose linear when acc_amplitude == 0 but
        # it was giving troubles with tensorflow, and since in our signals the only acc_amplitude=0  happens at the end,
        # our case is represented exactly. It will be necessary to rewrite this part as soon as signals with
        # acc_amplitude=0 in the middle will be considered.
        current = A_i_step_linear(t, i + 1, beta, 0) + 2 * beta * delta * A_i_step_squared(t, i, beta, 0)
        output += (current - previous) * ld.heaviside(t - period * i - phase, .5)

        return output

    return mu, tau


def mutau_delta(period, phase, amplitudes=[1, -1], library_derivatives='tensorflow'):
    ld = ld_operations(library_derivatives)

    def A_i_step(t, i, beta, acc_amplitude):
        current = ld.exp(2 * beta * acc_amplitude) * (t - period * i - phase)
        return current

    def mu(t):
        output = 0
        for i, amplitude in enumerate(amplitudes):
            output += amplitude * ld.heaviside(t - period * i - phase, .5)

        return output

    def tau(t, beta):
        output = t  # t
        previous = t - phase
        time_passed = phase

        for i, amplitude in enumerate(amplitudes):
            acc_amplitude = sum(amplitudes[:i + 1])
            current = A_i_step(t, i, beta, acc_amplitude)
            output += (current - previous) * ld.heaviside(t - period * i - phase, .5)

            time_passed += period
            current_evaluated = A_i_step(time_passed, i, beta, acc_amplitude)
            previous = current - current_evaluated

        return output

    return mu, tau


def ug_integrand(nu, factor, t, library_derivatives):
    ld = ld_operations(library_derivatives)
    nu = ld.abs(nu)
    return 1 / t / ld.gamma(ld.abs(nu)) * ld.exp(-factor / t) * ld.pow(factor / t, nu)


def isi(t, v0, sigma, beta, mu_, tau_, absolute=False, library_derivatives='tensorflow'):
    ld = ld_operations(library_derivatives)

    nu = tf.cast(tf.abs(1 / 2 / beta), dtype=tf.keras.backend.floatx())
    jacobian = ld.exp(2 * beta * mu_(t))
    tau = tau_(t, beta)
    v = v0 * ld.exp(mu_(tau))
    factor = v ** (-2 * beta) / (2 * sigma ** 2 * beta ** 2)
    g = jacobian * ug_integrand(
        nu, factor, tau,
        library_derivatives=library_derivatives)  # ug_integrand G

    if absolute:
        g = ld.abs(g)
    return g


def hit(t, v0, sigma, beta, mu_, tau_, library_derivatives='tensorflow'):
    ld = ld_operations(library_derivatives)

    nu = tf.cast(tf.abs(1 / 2 / beta), dtype=tf.keras.backend.floatx())

    tau = tau_(t, beta)
    jacobian = 1 #ld.exp(2 * beta * mu_(t))
    v = v0 * ld.exp(mu_(tau))
    factor = v ** (-2 * beta) / (2 * sigma ** 2 * beta ** 2 * tau)
    g = jacobian * tf.math.igammac(nu, factor, name=None)
    return g


def initialize_placeholders(library, placeholders_names='t q b v'):
    global library_derivatives
    library_derivatives = library

    if library in ['tensorflow', 'tf']:
        tf.keras.backend.set_floatx('float64')
        placeholders = [tf.keras.layers.Input((1,), name=n) for n in placeholders_names.split()]

    elif library in ['sympy', 'sp']:
        placeholders = sp.symbols(placeholders_names)

    else:
        raise NotImplementedError

    return placeholders


def signal_definition(study, q, amplitude_deltas=1, amplitude_steps=1, phase=1, n_params=1, vocab_size=1,
                      n_repetitions=1,                      language_length=1, library_derivatives='tensorflow'):
    ld = ld_operations(library_derivatives)

    if study == 'delta_amplitude':
        amplitudes = [q]
        mu_, tau_ = mutau_delta(period=0, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(-amplitude_deltas, amplitude_deltas, n_params)

    elif study == 'zero':
        amplitudes = [0]
        mu_, tau_ = mutau_delta(period=0, phase=q, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(-0, 0, n_params)

    elif study == 'delta_phase':
        amplitudes = [amplitude_deltas] #amplitude_deltas / 3
        mu_, tau_ = mutau_delta(period=0, phase=q, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)

    elif study == 'deltas_period':
        amplitudes = [-amplitude_deltas, amplitude_deltas]
        mu_, tau_ = mutau_delta(period=q, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)

    elif study == 'deltas_period_2':
        amplitudes = [amplitude_deltas, amplitude_deltas]
        mu_, tau_ = mutau_delta(period=q, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)

    elif study == 'steps_period':
        amplitudes = [amplitude_steps, amplitude_steps, amplitude_steps]
        mu_, tau_ = mutau_step(period=q, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)


    elif study == 'one_step':
        amplitudes = [amplitude_steps, -amplitude_steps]
        mu_, tau_ = mutau_step(period=q, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)

    elif study == 'steps_oscillation':
        amplitudes = [amplitude_steps] + np.tile(np.array([-2 * amplitude_steps, +2 * amplitude_steps]),
                                                 n_repetitions).tolist() + [-2 * amplitude_steps, amplitude_steps]
        mu_, tau_ = mutau_step(period=q, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)

    elif study == 'hmm_language':
        from hmmlearn import hmm
        np.random.seed(23)
        model = hmm.MultinomialHMM(n_components=vocab_size)  # regulier (Chomsky)
        startprob_ = np.random.rand(vocab_size)
        startprob_ /= np.sum(startprob_)
        model.startprob_ = startprob_

        transmat_, emissionprob_ = np.random.rand(vocab_size, vocab_size), np.random.rand(vocab_size, vocab_size)
        transmat_ /= np.sum(transmat_, axis=1)[..., None]
        emissionprob_ /= np.sum(emissionprob_, axis=1)[..., None]

        model.transmat_ = transmat_
        model.emissionprob_ = emissionprob_

        X, Z = model.sample(language_length)
        ws = amplitude_steps * np.random.randn(vocab_size)
        ws -= np.mean(ws)
        amplitudes = ws[Z]
        amplitudes[1:] = amplitudes[1:] - amplitudes[:-1]
        amplitudes = amplitudes.tolist()
        amplitudes.append(-ws[Z][-1])
        mu_, tau_ = mutau_step(period=q, phase=phase, amplitudes=amplitudes, library_derivatives=library_derivatives)
        q0s = np.linspace(phase, 5 * phase, n_params)

    elif study == 'gumbel_language':
        # performing the gumbel trick as here https://arxiv.org/pdf/1904.04079.pdf
        np.random.seed(seed=5)

        ws = 1.5 * amplitude_steps * np.random.rand(vocab_size)
        ws -= np.mean(ws)

        beta_gumbel, tau_gumbel = 1.1, .01
        logits = [(i + 1) ** (-q) for i in range(vocab_size)]
        ps = [l / ld.sum(logits) for l in logits]

        aux_amplitudes = []
        for _ in range(language_length):
            ns = np.random.rand(vocab_size)
            exps = [ld.exp((ld.log(p) + -beta_gumbel * ld.log(-ld.log(n))) / tau_gumbel) for p, n in zip(ps, ns)]
            z = [e / ld.sum(exps) for e in exps]

            a = ld.sum([z_i * w_i for z_i, w_i in zip(z, ws)])
            aux_amplitudes.append(a)

        amplitudes = [aux_amplitudes[0]] + [aux_amplitudes[i] - aux_amplitudes[i - 1]
                                            for i in range(1, language_length)] + [-aux_amplitudes[-1]]

        mu_, tau_ = mutau_step(period=phase / 2, phase=phase / 2, amplitudes=amplitudes,
                               library_derivatives=library_derivatives)
        q0s = np.linspace(.99, .3, n_params)

    else:
        raise NotImplementedError

    return mu_, tau_, q0s, amplitudes

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize

def solve_params(twitter_data, beta, alpha, p_verify, n_val, degree, pred_iterations):

    def calc_diff_eq(b, a, p, s_0, i_0, r_0, iter):
        s, i, r = [s_0], [i_0], [r_0]

        for it in range(iter-1):
            s_i, i_i, r_i = sir_model(s[it], i[it], r[it], b, a, p)
            s.append(s_i)
            i.append(i_i)
            r.append(r_i)

        return s, i, r

    def sir_model(S, I, R, b, a, p):

        n = S + I + R
        div = (I * (1 + a) + R * (1 - a))
        f = (degree / n) * b * ((I * (1 + a)) / div)
        g = (degree / n) * b * ((I * (1 - a)) / div)

        new_s = (1 - b) * S
        new_i = f * S + (1 - p) * I
        new_r = g * S + p * I + R

        return new_s, new_i, new_r
    def error_function(params):
        b, a, p, n = params

        new_S, new_I, new_R = calc_diff_eq(b, a, p, n - twitter_data[0], twitter_data[0], 0, len(twitter_data))

        error = np.mean((twitter_data - new_I[:len(twitter_data)]) ** 2)
        return error

    def const1(params):
        b, a, p, n = params
        return 1 - (degree / n) * b + 0.005

    # create minimization parameters
    init_params = np.array([beta, alpha, p_verify, n_val])

    bounds = ((0.0, float('inf')), (0.01, 0.99), (0.01, 0.99), (1, float('inf')))
    cons = {'type': 'ineq', 'fun': const1}
    #minimize the difference between the sir model and the twitter data
    result = minimize(error_function, init_params, bounds=bounds, constraints=cons)

    optimal_beta, optimal_alpha, optimal_p, n = result.x

    #calculate the sir model progression
    s, i, r = calc_diff_eq(optimal_beta, optimal_alpha, optimal_p, n, twitter_data[0], 0, pred_iterations)

    return_dict = {
        "beta": optimal_beta,
        "alpha": optimal_alpha,
        "p_verify": optimal_p,
        "s": s,
        "i": i,
        "r": r,
        "n": n
    }

    return return_dict

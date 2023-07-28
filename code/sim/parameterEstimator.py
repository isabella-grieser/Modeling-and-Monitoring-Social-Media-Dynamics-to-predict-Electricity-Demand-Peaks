import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize

def solve_params(s, i, r, start_time, end_time, time_step, twitter_data, beta, alpha, degree, p_verify):
    def sir_model(t, y, b, a, p, d):
        S, I, R = y
        n = S + I + R
        div = (I * (1 + a) + R * (1 - a))
        if div == 0:
            div = 1
        dsdt = - b * d / n * S
        didt = (b * d / n) * (I * (1 + a) / div) * S - p * I
        drdt = (b * d / n) * (R * (1 - a) / div) * S + p * I
        return [dsdt, didt, drdt]

    def error_function(params):
        b, a, p, d, S, I, R = params
        init = [S, I, R]
        solution = solve_ivp(lambda t, y: sir_model(t, y, b, a, p, d), time_span, init,
                             t_eval=time_points)
        i_pred = solution.y[1]
        error = np.mean((twitter_data - i_pred) ** 2)
        return error

    #create minimization parameters
    time_span = (start_time, end_time)
    time_points = np.arange(start_time, end_time, time_step)
    init_params = [beta, alpha, p_verify, degree, s, i, r]

    bounds = ((0, 1), (0, 1), (0, 1), (0, float('inf')), (0, float('inf')), (0, float('inf')), (0, float('inf')))
    #minimize the difference between the sir model and the twitter data
    result = minimize(error_function, init_params, bounds=bounds)
    optimal_beta, optimal_alpha, optimal_p, optimal_d, optimal_s, optimal_i, optimal_r = result.x

    init = [optimal_s, optimal_i, optimal_r]
    #calculate the sir model progression
    solution = solve_ivp(lambda t, y: sir_model(t, y, optimal_beta, optimal_alpha, optimal_p, optimal_d),
                         time_span, init, t_eval=time_points)

    return_dict = {
        "beta": optimal_beta,
        "alpha": optimal_alpha,
        "p_verify": optimal_p,
        "degree": optimal_d,
        "s_init": optimal_s,
        "i_init": optimal_i,
        "r_init": optimal_r,
        "s": solution.y[0],
        "i": solution.y[1],
        "r": solution.y[2]
    }

    return return_dict
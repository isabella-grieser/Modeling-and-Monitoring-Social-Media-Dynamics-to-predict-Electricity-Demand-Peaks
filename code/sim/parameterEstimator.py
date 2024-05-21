import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize

def solve_params(s, i, r, start_time, time_step, twitter_data, beta, alpha, degree_ratio, p_verify, end_time2):

    t_eval2 = np.arange(start_time, end_time2, time_step)
    def sir_model(t, y, b, a, p):
        S, I, R = y
        n = S + I + R
        d = degree_ratio * n
        div = (I * (1 + a) + R * (1 - a))
        if div == 0:
            div = 1
        dsdt = - 2 * b * d / n * S
        didt = 2 * (b * d / n) * (I * (1 + a) / div) * S - p * I
        drdt = 2 * (b * d / n) * (R * (1 - a) / div) * S + p * I
        return [dsdt, didt, drdt]

    def error_function(params):
        b, a, p,  S, I, R = params
        init = [S, I, R]
        solution = solve_ivp(lambda t, y: sir_model(t, y, b, a, p), (start_time, end_time2), init,
                             t_eval=t_eval2)
        i_pred = solution.y[1]
        error = np.mean((twitter_data - i_pred[:len(twitter_data)]) ** 2)
        return error

    #the constraint is written as an inequality to allow for rounding errors
    #rounding it and putting it as an equality does not work for the minimization algorithm
    def const(params):
        b, a, p, S, I, R = params
        init = [S, I, R]
        solution = solve_ivp(lambda t, y: sir_model(t, y, b, a, p), (start_time, end_time2), init,
                             t_eval=t_eval2)
        i_pred = solution.y[1]
        return 0.5 - i_pred[-1]

    #create minimization parameters
    init_params = np.array([beta, alpha, p_verify, s, i, r])

    bounds = ((0.01, 0.99), (0.01, 0.99), (0, 1), (1, float('inf')), (0, float('inf')), (0, float('inf')))

    cons = ({'type': 'ineq', 'fun': const})

    #minimize the difference between the sir model and the twitter data
    result = minimize(error_function, init_params, bounds=bounds, constraints=cons)

    optimal_beta, optimal_alpha, optimal_p, optimal_s, optimal_i, optimal_r = result.x
    init = [optimal_s, optimal_i, optimal_r]

    #calculate the sir model progression
    solution = solve_ivp(lambda t, y: sir_model(t, y, optimal_beta, optimal_alpha, optimal_p),
                         (start_time, end_time2), init,
                         t_eval=t_eval2)

    return_dict = {
        "beta": optimal_beta,
        "alpha": optimal_alpha,
        "p_verify": optimal_p,
        "s_init": optimal_s,
        "i_init": optimal_i,
        "r_init": optimal_r,
        "s": solution.y[0],
        "i": solution.y[1],
        "r": solution.y[2]
    }

    return return_dict
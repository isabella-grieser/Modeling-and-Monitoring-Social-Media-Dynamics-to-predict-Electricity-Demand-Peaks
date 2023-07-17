import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def calculate_params(s, i, r, n, start_time, end_time, time_step, twitter_data, beta, alpha, p_verify, degree):
    initial_conditions = [s, i, r]
    time_span = (start_time, end_time)
    time_points = np.arange(start_time, end_time, time_step)
    initial_params = np.array([beta, alpha, n])

    def error_function(params):
        b, a = params
        sol = solve_ivp(sir_model, time_span, initial_conditions, args=(b, a), t_eval=time_points)
        i_predicted = sol.y[1]
        error = np.mean((twitter_data - i_predicted) ** 2)
        return error

    def sir_model(t, y, b, a, N):
        S, I, R = y
        dSdt = - beta * S
        dIdt = (b * degree / N) * (R * (1 + a) / (I * (1 + a) + R * (1 - a))) * S - p_verify * I
        dRdt = (b * degree / N) * (R * (1 - a) / (I * (1 + a) + R * (1 - a))) * S + p_verify * I
        return [dSdt, dIdt, dRdt]

    result = minimize(error_function, initial_params, method='Nelder-Mead')

    optimal_params = result.x
    optimal_beta, optimal_alpha = optimal_params

    solution = solve_ivp(sir_model, time_span, initial_conditions, args=(optimal_beta, optimal_alpha),
                         t_eval=time_points)
    return solution.y



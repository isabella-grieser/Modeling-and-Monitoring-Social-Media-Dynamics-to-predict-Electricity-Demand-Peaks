
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Define the SIR model equations
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Define the error function
def error_function(params):
    beta, gamma = params
    solution = solve_ivp(sir_model, time_span, initial_conditions, args=(beta, gamma), t_eval=time_points)
    I_predicted = solution.y[1]  # Infected individuals (I)
    error = np.mean((I_data - I_predicted)**2)  # Mean squared error
    return error

# Set the initial conditions, time span, and data
initial_conditions = [initial_S, initial_I, initial_R]  # Replace with your own initial values
time_span = (start_time, end_time)  # Replace with your own start and end times
time_points = np.arange(start_time, end_time, time_step)  # Replace with your own time points
I_data = np.array([data1, data2, ...])  # Replace with your own observed infection data

# Set the initial guess for parameters
initial_params = [initial_beta, initial_gamma]  # Replace with your own initial guesses

# Optimize the parameters
result = minimize(error_function, initial_params, method='Nelder-Mead')

# Extract the optimized parameters
optimal_params = result.x
optimal_beta, optimal_gamma = optimal_params

# Numerically integrate the SIR model equations with the optimized parameters
solution = solve_ivp(sir_model, time_span, initial_conditions, args=(optimal_beta, optimal_gamma), t_eval=time_points)
S_values, I_values, R_values = solution.y

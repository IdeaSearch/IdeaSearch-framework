## pre definition
N=input()
# N = "temp_functions.py"
data_file = "/data/sonny/PDF/pdf_data_2_100.txt"
base_path = "/data/sonny/PDF/joint"
function_number = 4 # index of parton type (0: uv, 2: dv, 4:u, 6:d, 8:g, 10:s)
parton_map = {0:'uv', 2:'dv', 4:'u', 6:'d', 8:'g', 10:'s'}

import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import importlib.util

## Get Generated Functions
def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
island_module = load_module_from_file("island_module", f"{base_path}/{N}")

# Define the functions
def fuv(x,a0,a1,a2):
    return a0 * x ** (a1 - 1) * (1-x) ** (a2) * island_module.P_uv(x)

def fdv(x,a0,a1,a2):
    return a0 * x ** (a1 - 1) * (1-x) ** (a2) * island_module.P_dv(x) 
    
def fu(x,a0,a1,a2):
    return a0 * x ** (a1 - 1) * (1-x) ** (a2) * island_module.P_u(x) 
    
def fd(x,a0,a1,a2):
    return a0 * x ** (a1 - 1) * (1-x) ** (a2) * island_module.P_d(x)
    
def fg(x,a0,a1,a2):
    return a0 * x ** (a1 - 1) * (1-x) ** (a2) * island_module.P_g(x)
    
def fs(x,a0,a1,a2):
    return a0 * x ** (a1 - 1) * (1-x) ** (a2) * island_module.P_s(x)

# Make sure fuv and fdv is the first and the second
# Make sure fs is the last, because of sbar
funcs = [fuv, fdv, fu, fd, fg, fs]

def get_para_num(funcs):  # Get the number of parameters for each function
    para_num = []
    for func in funcs:
        sig = inspect.signature(func)
        para_num.append(len(sig.parameters) - 1)  # Subtract 1 to exclude 'x'
    return para_num

# Generate initial parameters
para_nums = get_para_num(funcs)
paras = [np.ones(para_num) for para_num in para_nums]  # Initial parameters as a list of arrays

## Generate PDF data
# 读取数据
def read_pdf_data():
    with open(data_file, 'r') as f:
        lines = f.readlines()
        data = [line.strip().split(',') for line in lines]
        data = [[float(item) for item in row] for row in data]
        
    return data
data = read_pdf_data()


# Calculate total momentum fraction
def total_momentum_fraction(params):
    # print("lambda: ", lambda_)
    integral_sum = 0.0
    start_idx = 0
    
    for i, func in enumerate(funcs):
        end_idx = start_idx + para_nums[i]
        func_params = params[start_idx:end_idx]
        start_idx = end_idx

        # Assuming each row of data has x value and then corresponding predicted values
        if i == len(funcs) - 1:
            integral_sum += quad(lambda x: x*func(x, *func_params), 0.0, 1.0, epsabs=1e-3, epsrel=1e-2)[0] * 2
        else:
            integral_sum += quad(lambda x: x*func(x, *func_params), 0.0, 1.0, epsabs=1e-3, epsrel=1e-2)[0]

    return integral_sum

def objective_function_joint(params, lambda_=None):
    # print("lambda: ", lambda_)
    total_error = 0.0
    start_idx = 0
    
    for i, func in enumerate(funcs):
        end_idx = start_idx + para_nums[i]
        func_params = params[start_idx:end_idx]
        start_idx = end_idx

        # Assuming each row of data has x value and then corresponding predicted values
        for j in range(len(data)):
            y_predicted = func(data[j][0], *func_params)*data[j][0] # data is x times pdf

            # Assuming error is computed based on actual vs. predicted values with weighting if needed
            total_error += ((data[j][2*i+1] - y_predicted) / data[j][2+2*i])**2
        
        if lambda_ is not None and i <= 1:
            # Add the momentum conservation constraint to the objective function
            integral_sum = quad(lambda x: func(x, *func_params), 0.0, 1.0, epsabs=1e-3, epsrel=1e-2)[0]
            total_error += lambda_ * (integral_sum - 1)**2 if i==1 else 0
            total_error += lambda_ * (integral_sum - 2)**2 if i==0 else 0
            
    if lambda_ is not None:
        # Add the momentum conservation constraint to the objective function
        total_error += lambda_ * (total_momentum_fraction(params) - 1)**2

    return total_error

# Function to optimize parameters for all functions together
def optimize_joint(initial_guess=None):
    if initial_guess is None:
        # Initialize with the mean of all parameter values
        initial_guess = np.concatenate(paras).tolist()
    bounds = [(None, None)] * len(initial_guess)
    
    # result = minimize(objective_function_joint, initial_guess, args=(10,), method="L-BFGS-B", bounds=bounds)
    result = minimize(objective_function_joint, initial_guess, method="L-BFGS-B", bounds=bounds)
    print("momentum conservation result: ", total_momentum_fraction(result.x))
    print("the optimized objective function value is: ", result.fun)
    print("Optimized parameters (joint):", result.x)
    print("输出结果：", -result.fun, end='', sep='')
    return result.x

# User choice for optimization
def main():
    
    # use optimize_separate firstly, then use optimize_joint
    # optimized_params = optimize_separate()
    # print("optimized params from separate: ", optimized_params)
    optimized_params = optimize_joint()
    # print("optimized params from joint: ", optimized_params)

    npdata = np.asarray(data)

    # Plotting the fit
    x_plot = np.logspace(-7, -0.1, 400)
    
    plt.figure(figsize=(12, 6))
    
    start_idx = 0
    for i, func in enumerate(funcs):
        end_idx = start_idx + para_nums[i]
        func_params = optimized_params[start_idx:end_idx]
        start_idx = end_idx
        
        plt.errorbar(npdata[:, 0], npdata[:, 2*i+1], yerr=npdata[:, 2*i+2], fmt='o', label=f'x*pdf for {parton_map[2*i]}')
        plt.semilogx(x_plot, x_plot*func(x_plot, *func_params), '-', label=f'Fit {parton_map[2*i]}')
    
    plt.xlim(1e-7,0.99)
    plt.ylim(-2,6)
    plt.legend()
    plt.title('Fitted Curves')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('fit_result.png')
    # plt.show()

if __name__ == "__main__":
    main()

# ... existing code...

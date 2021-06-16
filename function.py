# Write the functions here
import numpy as np

def func1(x): 
    return 2*np.power(x,2) 

def func2(x):
    return np.sin(x)

def func3(x):
    return np.power(x,4) + 3*np.power(x,2) + 1

def func4(x):
    return 0.2 + 0.4*np.power(x,2) + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)

def func5(x):
    return np.power(x,3) - 3*x + 4

f = [func1, func2, func3, func4, func5]
labels = ["2x^2", "sinx", "x^4 + 3x^2 + 1", "0.2 + 0.4x^2 + 0.3xsin(15x) + 0.05*cos(50x)", "x^3 - 3x + 4"]
lower = [-10, -10, -10, 0, -1]
upper = [10, 10, 10, 1, 1]
epoch = [5, 15, 5, 15, 500]
batch_Size = [1, 1, 1, 1, 1]
# validation split
zoom_factor = [1, 1, 1, 1, 1]
div_factor = [10, 10, 10, 1000, 1000]




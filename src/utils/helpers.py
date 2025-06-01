import math

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def linear(x):
    return x

def linear_derivative(x):
    return 1

def tanh_func(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x)**2

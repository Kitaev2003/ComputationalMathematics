import numpy as np
import matplotlib.pyplot as plt
import math as mth

# sin(x^2)
def func1(x):
    return mth.sin(x * x)

def func1Diff(x):
    return 2 * x * mth.cos(x * x)

# cos(sin(x))
def func2(x):
    return mth.cos(mth.sin(x))

def func2Diff(x):
    return -1 * mth.cos(x) * mth.sin(mth.sin(x))

# e^(sin(cos(x)))
def func3(x):
    return mth.exp(mth.sin(mth.cos(x)))

def func3Diff(x):
    return -1 * func3(x) * mth.sin(x) * mth.cos(mth.cos(x))

# ln(x+3)
def func4(x):
    return mth.log(x + 3)

def func4Diff(x):
    return 1 / (x + 3)

# sqrt(x+3)
def func5(x): 
    return mth.sqrt(x + 3)

def func5Diff(x):
    return 1 / (2 * func5(x))

# f(x+h) - f(x)
# —————————————
#       h
def diff1(f, x, h) :
    return (f(x + h) - f(x)) / h

# f(x) - f(x-h)
# —————————————
#       h
def diff2(f, x, h) :
    return (f(x) - f(x - h)) / h

# f(x+h) - f(x-h)
# ———————————————
#       2h
def diff3(f, x, h) :
    return (f(x + h) - f(x - h)) / (2 * h)

# 4 f(x+h) - f(x-h)   1 f(x+2h) - f(x-2h) 
# — ——————————————— - — ——————————————————
# 3        2h         3        4h
def diff4(f, x, h) :
    return (4/3) * ((f(x + h) - f(x - h)) / (2 * h)) \
         - (1/3) * ((f(x + 2 * h) - f(x - 2 * h)) / (4 * h))

# 3 f(x+h) - f(x-h)   3 f(x+2h) - f(x-2h)   1  f(x+3h) - f(x-3h)
# — ——————————————— - — ————————————————— + —— ——————————————————
# 2        2h         5        4h           10         6h
def diff5(f, x, h) :
    return (3 / 2) * ((f(x + h) - f(x - h)) / (2 * h)) \
         - (3 / 5) * ((f(x + 2 * h) - f(x - 2 * h)) / (4 * h)) \
         + (1 / 10) * ((f(x + 3 * h) - f(x - 3 * h)) / (6 * h))

# delta = |A - Ax|
def absolute_error(diff_false, diff_true):
    return abs(diff_false - diff_true)

n = [i for i in range (0, 21)]
h = [2/(2 ** i) for i in range(0, 21)]

# List with any function
DiffFunc = [diff1, diff2, diff3, diff4, diff5]
Function  = [func1, func2, func3, func4, func5]
DiffFunction = [func1Diff, func2Diff, func3Diff, func4Diff, func5Diff]

# The point at which we differentiate
num = 10

#Number function
number = 0

# Set plot
fig, axs = plt.subplots(5)

# Title
fig.suptitle('Function number 1')

for i in range(len(DiffFunc)):
    methods = DiffFunc[i]
    functionDiff = DiffFunction[number]
    PlotFunc = [absolute_error(methods(Function[number], num, h[i]), functionDiff(num)) for i in range (len(h))]
    # Make plot
    axs[i].plot(n, PlotFunc)
    # Add gridlines to the plot
    axs[i].grid(b=True)

    axs[i].set(xlabel='n-parametr', ylabel='Method ' + str(i), )
# Save plot
plt.savefig('Function' + str(number  + 1) + '.jpg', dpi = 500)

# Show plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math 

# Первая система
def F1(x):
    return np.array ([np.sin (x[0] + 1) - x[1] - 1.2, 2 * x[0] + np.cos (x[1]) - 2])

# Якобиан первой системы
def J1(x):
    jac = np.zeros ((2, 2))

    jac[0, 0] = np.cos (x[0] + 1)
    jac[0, 1] = -1
    jac[1, 0] = 2
    jac[1, 1] = -np.sin (x[1])

    return jac

# F(x) = 0 - заданное уравнение
F = lambda x: x * math.pow(2, x) - 1

# f(x) = x - приведенное уравнение
f = lambda x: math.pow(2, -x)

# f'(x) - производная 
f_der = lambda x: (-np.log(2)) / math.pow(2, x)

# График F(x)
plt.figure()
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=100)
ax1.set_title("F(x)")

x = np.arange(-2, 2, 0.01)
y = np.arange(-2, 2, 0.01)
for i in range (400):
    y[i] = F(x[i])

plt.grid()
plt.plot(x, y)
plt.savefig("Image1.jpg", dpi=500)

# График f'(x)

plt.figure()
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=100)
ax1.set_title("f'(x)")

x = np.arange(0.5, 1, 0.01)
y = np.arange(0.5, 1, 0.01)
for i in range (50):
    y[i] = f_der(x[i])

plt.grid()
plt.plot(x, y)
plt.savefig("Image2.jpg", dpi=500)

err = 1e-15            #точность
x_prev = 0             # x_n
x = 0.5                # x_(n+1)
while np.abs (x - x_prev) >= err:
    x_prev = x
    x = f(x_prev)

print ("Корень: ", x)
print ("Значение в точке:", F(x))


x_prev = np.array ([0, 0])        # x_n
x = np.array ([0.4, -0.3])        # x_(n+1)

while np.linalg.norm(x - x_prev) >= err:
    x_prev = x
    x = x_prev - np.matmul(np.linalg.inv(J1(x)), F1(x)) 

print ("Решение:", x)
print ("Значения F(x) в точке решения:", F1(x))

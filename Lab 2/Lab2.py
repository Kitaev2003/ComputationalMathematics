import scipy
import matplotlib.pyplot as plot
import numpy as np
import math as mth

def gacobi(A, b, N=40, x=None):                                                                                                                                                         
    if x is None:
        x = np.zeros(len(A[0]))

    data = np.zeros(N)                                                                                                                                                                    
    D = np.diag(A)
    T = A - np.diagflat(D)
                                                                                                                                                                          
    for i in range(N):
        x = (b - np.dot(T,x)) / D   
        diff = b - np.matmul(A, x)
        data[i] = norm(diff)  

    return x, data


def gaussSeidel(A, b, N=25, x=None):
    
    if x is None:
        x = np.zeros(len(A[0]))

    data = np.zeros(N)

    L = np.tril(A)
    U = A - L
    for i in range(N):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        diff = b - np.matmul(A, x)
        data[i] = norm(diff)

    return x, data

def upperRelaxation(A, b, w, N=20, x=None):
    
    if x is None:
        x = np.zeros(len(A[0]))

    data = np.zeros(N)

    u = np.triu(A)
    l = np.tril(A)
    L = A - u
    D = l + u - A
    U = A - l

    B = - np.matmul(np.linalg.inv(D + w * L), (w - 1) * D + w * U)
    F =   np.matmul(np.linalg.inv(D + w * L), b) * w

    for i in range(N):
        x = np.matmul(B, x) + F
        diff = b - np.matmul(A, x)
        data[i] = norm(diff)

    return x, data

def gauss(A,b):
    n = len(A)
    M = A

    M = np.hstack((M,np.array([b]).T))

    for i in range(n):

        leading = i + np.argmax(np.abs(A[:,i][i:]))
        M[[i, leading]] = M[[leading, i]] 

        M[i] /= M[i][i]
        row = M[i]

        for r in M[i + 1:]:
            r -= r[i] * row

    for i in range(n - 1, 0, -1):
        row = M[i]
        for r in reversed(M[:i]):
            r -= r[i] * row

    return M[:,-1]

def LUmethod(A, f):
    P, L, U = scipy.linalg.lu(A)
    y = gauss(L, f)
    x = gauss(U, y)
    return x

def norm(vec):
  return np.max(np.abs(vec))

def matrixElem(i: int, j: int):
  if i == 99:
    return 1
  else:
    if i == j:
      return 10
    elif i == (j + 1) or j == (i + 1):
      return 1
    else: 
      return 0

def getLDU(matrix):
  u = np.triu(matrix)
  l = np.tril(matrix)

  lower = matrix - u
  diagonal = l + u - matrix
  upper = matrix - l

  return lower, diagonal, upper


def getMatrix():
  return np.fromfunction(np.vectorize(matrixElem), (100, 100), dtype=np.double)

def bElem(i: int):
  return (i + 1)

def getB():
  return np.fromfunction(np.vectorize(bElem), (100,), dtype = np.double)

def makePlot(title, data):
    plot.figure(figsize=(15, 15))

    plot.xlabel("iteration number")
    plot.ylabel("log(abs_accurancy)")
    plot.yscale("log")

    plot.title(title)
    iterations = [i for i in range(len(data))]
    plot.plot(iterations, data, ".-")
    plot.show()

def plotFew(title, names, datas):
    plot.figure(figsize=(15, 15))

    plot.xlabel("iteration number")    
    plot.ylabel("log(abs_accurancy)")
    plot.yscale("log")

    plot.title(title)
    for ind in range(len(datas)):
        iterations = [i for i in range(len(datas[ind]))]
        plot.plot(iterations, datas[ind], ".-", label=names[ind])
        plot.legend()
    plot.show()


eps = 1e-6

mat = getMatrix()
b = getB()

x = gauss(mat, b)
x = LUmethod(mat, b)

A = np.array([[4.0,-1.0,1.0],[2.0,5.0,2.0],[1.0,2.0,4.0]])
b = np.array([8.0,3.0,11.0])

x, data = gacobi(A, b)
makePlot("gacobi", data)

x, data = gaussSeidel(A, b)
makePlot("gauss", data)

data_for_relaxation = []
names = []
for w in np.arange(1., 2.2, 0.2):
    w = mth.ceil(w * 10) / 10
    x, data = upperRelaxation(A, b, w)
    data_for_relaxation.append(data)
    names.append("w=" + str(w))

plotFew("upperRelaxation", names, data_for_relaxation)
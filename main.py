import numpy as np
from numpy.linalg import norm

H = [[-3, 2.25],
     [2.25, -4]]

def calcStep(grad):
     top = (np.matmul(np.transpose(grad), grad))
     bottom = np.matmul((np.matmul(np.transpose(grad), H)), grad)
     return top / bottom


def calcGrad(x1, x2):
     grad = [2.25*x2 - 3*x1, 2.25*x1 + 1.75 - 4*x2]
     return grad

def calcNextX(x1, x2):
     grad = calcGrad(x1, x2)
     step = calcStep(grad)
     print("Step size:", str(step))
     print("The l2 norm of the gradient vector:", norm(grad))
     newX = (np.array([x1, x2]) - np.array(grad) * step)
     return newX

def main():
     print("----------------------------------------")
     print("Start with Coordinate Pair: [1, 1]")
     print("----------------------------------------")
     print("Iteration 1")
     newX = calcNextX(1, 1)
     print("Coordinate Pair:", str(newX))
     count = 2
     print("----------------------------------------")
     while (round(newX.item(0),  4) != round(21/37,  4) or round(newX.item(1), 4) != round(28/37,  4)):
          print("Iteration", str(count))
          newX = calcNextX(newX.item(0), newX.item(1))
          print("Coordinate Pair:", str(newX))
          count += 1
          print("----------------------------------------")

main()
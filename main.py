import os
from nn import Perceptron

def check(x):
    if (x != 0 and x != 1):
        print('Please the values must be 0 or 1')
        return False
    return True

nn = Perceptron()
nn.learn(n=50, save=True)

while True:
    print('Input first value:')
    x = int(input())
    if not check(x): continue

    print('Input second value:')
    y = int(input())
    if not check(x): continue

    print(x, "or", y, "is : ", nn.perceptron(x,y))

    if input('Continue? (y, n) ') == 'n': exit()

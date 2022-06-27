import os
from nn import Perceptron

percep_nn = Perceptron()
percep_nn.learn(n=50, save=True)

while True:
    # Testing phase
    print('Input first value:')
    x = int(input())
    print('Input second value:')
    y = int(input())
    weights = percep_nn.weights
    outputP = x * weights[0] + y * weights[1] + percep_nn.bias * weights[2]
    if outputP > 0:  # activation function
        outputP = 1
    else:
        outputP = 0
    print(x, "or", y, "is : ", outputP)

    if input('Continue? (y, n) ') == 'n': exit()

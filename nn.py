import numpy, random, os.path
import numpy as np


class Perceptron:
    lr = 1  # learning rate
    bias = 1  # value of bias

    weights = numpy.array([random.random(), random.random(),
                           random.random()])  # weights generated in a list (3 weights in total for 2 neurons and the bias)

    function = 'h' #activation function 'h' (heaviside) or 's' (sigmoid)

    def set_parameters(self, lr, bias, weights, function):
        self.lr = lr
        self.bias = bias
        self.weights = weights

        if function != 'h' and function != 's':
            raise Exception("The function must be 'h' (heaviside) or 's' (sigmoid)")

    def core(self, input1, input2, output):
        outputP = input1 * self.weights[0] + input2 * self.weights[1] + self.bias * self.weights[2]
        if self.function == 'h':
            if outputP > 0:  # activation function (here Heaviside)
                outputP = 1
            else:
                outputP = 0
        elif self.function == 's':
            outputP = 1 / (1 + numpy.exp(-outputP))  # sigmoid function

        error = output - outputP
        self.weights[0] += error * input1 * self.lr
        self.weights[1] += error * input2 * self.lr
        self.weights[2] += error * self.bias * self.lr

    def learn(self, n=50, save=False):
        if save:
            if os.path.exists("Data/%dtrained.csv" % n):
                self.weights = np.loadtxt("Data/%dtrained.csv" % n)

        for i in range(n):
            self.core(1, 1, 1)  # True or true
            self.core(1, 0, 1)  # True or false
            self.core(0, 1, 1)  # False or true
            self.core(0, 0, 0)  # False or false

        if save:
            numpy.savetxt("Data/%dtrained.csv" % n, self.weights, delimiter=',')

    def perceptron(self, x, y):
        outputP = x * self.weights[0] + y * self.weights[1] + self.bias * self.weights[2]
        return 1 if outputP > 0 else 0


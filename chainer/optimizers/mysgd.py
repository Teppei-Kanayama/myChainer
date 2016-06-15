from chainer import cuda
from chainer import optimizer

import sys
sys.path.append("/home/mil/kanayama/chainer/chainer/")
import myoptimizer


class SGD(myoptimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        print("this instance is generated by mysgd.py")
        super(SGD, self).__init__()
        self.lr = lr

    def update_one_cpu(self, param, state):
        param.data -= self.lr * param.grad

    def update_one_gpu(self, param, state):
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(param.grad, self.lr, param.data)

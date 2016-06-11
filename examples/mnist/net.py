import chainer
import chainer.functions as F
import chainer.links as L


class Net1(chainer.Chain):
    insize = 32
    
    def __init__(self):
        super(Net1, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=1),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(1024, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 10),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        print self.conv1.W.data
        self.clear()
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h

class Net2(chainer.Chain):
    inseize = 32

    def __init__(self):
        super(Net2, self).__init__(
            conv1 = F.Convolution2D(3, 32, 3),
            bn1 = F.BatchNormalization(32),
            conv2 = F.Convolution2D(32, 64, 3, pad=1),
            bn2 = F.BatchNormalization(64),
            conv3 = F.Convolution2D(64, 64, 3, pad=1),
            fl4 = F.Linear(1024, 256),
            fl5 = F.Linear(256, 10)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 2)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)
        h = F.dropout(F.relu(self.fl4(h)), train=self.train)
        h = self.fl5(h)
        return h

class MnistMLPParallel(chainer.Chain):

    """An example of model-parallel MLP.

    This chain combines four small MLPs on two different devices.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLPParallel, self).__init__(
            first0=MnistMLP(n_in, n_units // 2, n_units).to_gpu(0),
            first1=MnistMLP(n_in, n_units // 2, n_units).to_gpu(1),
            second0=MnistMLP(n_units, n_units // 2, n_out).to_gpu(0),
            second1=MnistMLP(n_units, n_units // 2, n_out).to_gpu(1),
        )

    def __call__(self, x):
        # assume x is on GPU 0
        x1 = F.copy(x, 1)

        z0 = self.first0(x)
        z1 = self.first1(x1)

        # sync
        h0 = z0 + F.copy(z1, 0)
        h1 = z1 + F.copy(z0, 1)

        y0 = self.second0(F.relu(h0))
        y1 = self.second1(F.relu(h1))

        # sync
        y = y0 + F.copy(y1, 0)
        return y

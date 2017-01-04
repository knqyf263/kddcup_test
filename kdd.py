import chainer
import chainer.links as L
import chainer.functions as F

class Kdd(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(Kdd, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        y = self.l2(h1)
        return y

class Kdd2(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(Kdd2, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y

class KddRelu(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(KddRelu, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y

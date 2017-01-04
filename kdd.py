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
        h = self.l1(x)
        y = self.l2(h)
        return y

class Kdd2(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(Kdd2, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        y = self.l3(h)
        return y

class KddRelu(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(KddRelu, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y

class KddReluDropout(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(KddReluDropout, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h = F.dropout(F.relu(self.l1(x)))
        y = self.l2(h)
        return y

class KddBnormReluDropout(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(KddBnormReluDropout, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            bnorm1 = L.BatchNormalization(hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h = F.dropout(F.relu(self.bnorm1(self.l1(x))))
        y = self.l2(h)
        return y

class KddReluBnormDropout(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(KddReluBnormDropout, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            bnorm1 = L.BatchNormalization(hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h = F.dropout(self.bnorm1(F.relu(self.l1(x))))
        y = self.l2(h)
        return y

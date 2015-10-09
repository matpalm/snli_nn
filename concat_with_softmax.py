import math
import theano.tensor as T
import util
from updates import vanilla, rmsprop

class ConcatWithSoftmax(object):
    def __init__(self, states, n_labels, n_hidden_previous):
        self.input = T.concatenate(states)
        input_size = len(states) * n_hidden_previous
        # input -> hidden (sized somwhere between size of input & softmax)
        n_hidden = int(math.sqrt(input_size * n_labels))
        self.Wih = util.sharedMatrix(input_size, n_hidden, 'Wih')
        self.bh = util.shared(util.zeros((1, n_hidden)), 'bh')
        # hidden -> softmax
        self.Whs = util.sharedMatrix(n_hidden, n_labels, 'Whs')
        self.bs = util.shared(util.zeros((1, n_labels)), 'bs')
        self.params = [self.Wih, self.bh, self.Whs, self.bs]

    def params_for_l2_penalty(self):
        return self.params

    def updates_wrt_cost(self, cost, learning_rate):
        gradients = T.grad(cost=cost, wrt=self.params)
        return vanilla(self.params, gradients, learning_rate)

    def prob_pred(self):
        hidden = T.nnet.sigmoid(T.dot(self.input, self.Wih) + self.bh)
        prob_y = T.nnet.softmax(T.dot(hidden, self.Whs) + self.bs)
        pred_y = T.argmax(prob_y, axis=1)
        return (prob_y, pred_y)


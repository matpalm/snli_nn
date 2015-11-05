from dropout import dropout
import math
import theano
import theano.tensor as T
import util
from updates import vanilla, rmsprop

class ConcatWithSoftmax(object):
    def __init__(self, states, n_labels, n_hidden_previous, update_fn,
                 training=None, keep_prob=None):
        self.input = T.concatenate(states)
        if training is not None:
            self.input = dropout(self.input, training, keep_prob)
        input_size = len(states) * n_hidden_previous

        # input -> hidden (sized somwhere between size of input & softmax)
        n_hidden = int(math.sqrt(input_size * n_labels))
        self.Wih = util.sharedMatrix(input_size, n_hidden, 'Wih')
        self.bh = util.shared(util.zeros((1, n_hidden)), 'bh')
        # hidden -> softmax
        self.Whs = util.sharedMatrix(n_hidden, n_labels, 'Whs')
        self.bs = util.shared(util.zeros((1, n_labels)), 'bs')

        self.update_fn = globals().get(update_fn)
        if self.update_fn is None:
            raise Exception("no such update function", update_fn)

    def name(self):
        return "concat_with_softmax"

    def dense_params(self):
        return [self.Wih, self.bh, self.Whs, self.bs]

    def params_for_l2_penalty(self):
        return self.dense_params()

    def updates_wrt_cost(self, cost, learning_rate):
        gradients = util.clipped(T.grad(cost=cost, wrt=self.dense_params()))
        return self.update_fn(self.dense_params(), gradients, learning_rate)

    def prob_pred(self):
        hidden = T.nnet.sigmoid(T.dot(self.input, self.Wih) + self.bh)
        prob_y = T.nnet.softmax(T.dot(hidden, self.Whs) + self.bs)
        pred_y = T.argmax(prob_y, axis=1)
        return (prob_y, pred_y)


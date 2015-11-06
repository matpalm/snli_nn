import numpy as np
import theano
import theano.tensor as T
from updates import vanilla, rmsprop
import util

class SimpleRnn(object):
    def __init__(self, name, input_dim, hidden_dim, opts, update_fn, h0, inputs):
        self.name_ = name
        self.update_fn = update_fn
        self.h0 = h0
        self.inputs = inputs

        # hidden -> hidden
        self.Whh = util.sharedMatrix(hidden_dim, hidden_dim, 'Whh', orthogonal_init=True)

        # embedded input -> hidden
        self.Whe = util.sharedMatrix(hidden_dim, input_dim, 'Whe', orthogonal_init=True)

        # bias
        self.Wb = util.shared(util.zeros((hidden_dim,)), 'Wb')

    def name(self):
        return self.name_

    def dense_params(self):
        return [self.Whh, self.Whe, self.Wb]

    def params_for_l2_penalty(self):
        return self.dense_params()

    def updates_wrt_cost(self, cost, learning_rate):
        gradients = util.clipped(T.grad(cost=cost, wrt=self.dense_params()))
        return self.update_fn(self.dense_params(), gradients, learning_rate)

    def recurrent_step(self, inp, h_t_minus_1):
        h_t = T.tanh(T.dot(self.Whh, h_t_minus_1) +
                     T.dot(self.Whe, inp) +
                     self.Wb)
        return [h_t, h_t]

    def all_states(self):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[self.inputs],
                                     outputs_info=[self.h0, None])
        return h_t

    def final_state(self):
        return self.all_states()[-1]

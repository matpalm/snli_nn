import numpy as np
import theano
import theano.tensor as T
from updates import vanilla, rmsprop
import util

class SimpleRnn(object):
    def __init__(self, name, input_dim, hidden_dim, opts, update_fn, h0, inputs,
                 context=None, context_dim=None):
        self.name_ = name
        self.update_fn = update_fn
        self.h0 = h0
        self.inputs = inputs    # input sequence
        self.context = context  # additional context to add at each timestep of input

        # hidden -> hidden
        self.Uh = util.sharedMatrix(hidden_dim, hidden_dim, 'Uh', orthogonal_init=True)

        # embedded input -> hidden
        self.Wh = util.sharedMatrix(hidden_dim, input_dim, 'Wh', orthogonal_init=True)

        # context -> hidden (if applicable)
        if self.context:
            self.Whc = util.sharedMatrix(hidden_dim, context_dim, 'Wch',
                                         orthogonal_init=True)

        # bias
        self.bh = util.shared(util.zeros((hidden_dim,)), 'bh')

    def name(self):
        return self.name_

    def dense_params(self):
        params = [self.Uh, self.Wh, self.bh]
        if self.context:
            params.append(self.Whc)
        return params

    def params_for_l2_penalty(self):
        return self.dense_params()

    def updates_wrt_cost(self, cost, learning_rate):
        gradients = util.clipped(T.grad(cost=cost, wrt=self.dense_params()))
        return self.update_fn(self.dense_params(), gradients, learning_rate)

    def recurrent_step(self, inp, h_t_minus_1):
        h_t = (T.dot(self.Uh, h_t_minus_1) +
               T.dot(self.Wh, inp) +
               self.bh)
        if self.context:
            h_t += T.dot(self.Whc, self.context)
        h_t = T.tanh(h_t)
        return [h_t, h_t]

    def all_states(self):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[self.inputs],
                                     outputs_info=[self.h0, None])
        return h_t

    def final_state(self):
        return self.all_states()[-1]

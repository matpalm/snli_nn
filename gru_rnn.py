import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class GruRnn(object):
    def __init__(self, name, input_dim, hidden_dim, opts, update_fn, h0, inputs):
        self.name_ = name
        self.update_fn = update_fn
        self.h0 = h0
        self.inputs = inputs

        # params for standard recurrent step
        self.Uh = util.sharedMatrix(hidden_dim, hidden_dim, 'Uh', orthogonal_init=True)
        self.Wh = util.sharedMatrix(hidden_dim, input_dim, 'Wh', orthogonal_init=True)
        self.bh = util.shared(util.zeros((hidden_dim,)), 'bh')

        # params for reset gate; initial bias to not reset
        self.Ur = util.sharedMatrix(hidden_dim, hidden_dim, 'Ur', orthogonal_init=True)
        self.Wr = util.sharedMatrix(hidden_dim, input_dim, 'Wr', orthogonal_init=True)
        self.br = util.shared(np.asarray([opts.gru_initial_bias]*hidden_dim), 'br')

        # params for carry gate; initial bias to never carry h_t_minus_1
        self.Uz = util.sharedMatrix(hidden_dim, hidden_dim, 'Uz', orthogonal_init=True)
        self.Wz = util.sharedMatrix(hidden_dim, input_dim, 'Wz', orthogonal_init=True)
        self.bz = util.shared(np.asarray([opts.gru_initial_bias]*hidden_dim), 'bz')

    def name(self):
        return self.name_

    def dense_params(self):
        return [self.Uh, self.Wh, self.bh, 
                self.Ur, self.Wr, self.br, 
                self.Uz, self.Wz, self.bz]

    def params_for_l2_penalty(self):
        return self.dense_params()

    def updates_wrt_cost(self, cost, learning_rate):
        gradients = util.clipped(T.grad(cost=cost, wrt=self.dense_params()))
        return self.update_fn(self.dense_params(), gradients, learning_rate)

    def recurrent_step(self, inp, h_t_minus_1):
        # reset gate; how much will we zero out h_t_minus_1 in our candidate
        # next hidden state calculation?
        r = T.nnet.sigmoid(T.dot(self.Ur, h_t_minus_1) +
                           T.dot(self.Wr, inp) +
                           self.br)
        # candidate hidden state
        h_t_candidate = T.tanh(r * T.dot(self.Uh, h_t_minus_1) +
                               T.dot(self.Wh, inp) +
                               self.bh)
        # carry gate; how much of h_t_minus_1 will we take with h_candidate?
        z = T.nnet.sigmoid(T.dot(self.Uz, h_t_minus_1) +
                           T.dot(self.Wz, inp) +
                           self.bz)
        # actual hidden state affine combo of last state and candidate state
        h_t = (1 - z) * h_t_minus_1 + z * h_t_candidate
        return [h_t, h_t]

    def all_states(self):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[self.inputs],
                                     outputs_info=[self.h0, None])
        return h_t

    def final_state(self):
        return self.all_states()[-1]


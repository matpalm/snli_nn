import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class SimpleRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden, orthogonal_init, idxs, forwards, Wx=None):
        if Wx is None:
            self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx', orthogonal_init=orthogonal_init)
        else:
            self.Wx = Wx
        self.Whh = util.sharedMatrix(n_hidden, n_hidden, 'Whh', orthogonal_init=orthogonal_init)
#        self.Whh = util.shared(util.eye(n_hidden), 'Whh')
        self.Whe = util.sharedMatrix(n_hidden, n_embedding, 'Whe', orthogonal_init=orthogonal_init)
        self.Wb = util.shared(util.zeros((n_hidden,)), 'Wb')
        self.sequence_embeddings = self.Wx[idxs]
        self.forwards = forwards

    def params(self):
        return [self.sequence_embeddings, self.Whh, self.Whe, self.Wb]

    def updates_wrt_cost(self, cost, learning_rate, existing_updates):
        # calculate dense updates
        params = [self.Whh, self.Whe, self.Wb]
        gradients = T.grad(cost=cost, wrt=params)
        updates = vanilla(params, gradients, learning_rate)
        # calculate a sparse update for embeddings (IF REQUIRED)
        # this is clumsy; we want tied weights & updates per layer... hmm...
        existing_update_params = [p for (p, u) in existing_updates]
        if self.Wx not in existing_update_params:
            gradient = T.grad(cost=cost, wrt=self.sequence_embeddings)
            updates += [(self.Wx, T.inc_subtensor(self.sequence_embeddings, -learning_rate * gradient))]
        return updates

    def recurrent_step(self, embedding, h_t_minus_1):
        h_t = T.tanh(T.dot(self.Whh, h_t_minus_1) + T.dot(self.Whe, embedding) + self.Wb)
        return [h_t, h_t]

    def final_state_given(self, h0):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     go_backwards=not self.forwards,
                                     sequences=[self.sequence_embeddings],
                                     outputs_info=[h0, None])
        return h_t[-1]

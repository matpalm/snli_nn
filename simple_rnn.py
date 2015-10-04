import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class SimpleRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden, orthogonal_init):
        self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx', orthogonal_init=orthogonal_init)
        self.Whh = util.sharedMatrix(n_hidden, n_hidden, 'Whh', orthogonal_init=orthogonal_init)
#        self.Whh = util.shared(util.eye(n_hidden), 'Whh')
        self.Whe = util.sharedMatrix(n_hidden, n_embedding, 'Whe', orthogonal_init=orthogonal_init)
        self.Wb = util.shared(util.zeros((n_hidden,)), 'Wb')

    def set_idxs(self, idxs):
        self.sequence_embeddings = self.Wx[idxs]

    def updates_wrt_cost(self, cost, learning_rate):
        # calculate dense updates
        params = [self.Whh, self.Whe, self.Wb]
        gradients = T.grad(cost=cost, wrt=params)
        updates = vanilla(params, gradients, learning_rate)
        # calculate a sparse update for embeddings
        gradient = T.grad(cost=cost, wrt=self.sequence_embeddings)
        updates += [(self.Wx, T.inc_subtensor(self.sequence_embeddings, -learning_rate * gradient))]
        return updates

    def recurrent_step(self, embedding, h_t_minus_1):
        # calc new hidden state; elementwise add of embedded input &
        # recurrent weights dot _last_ hiddenstate
        h_t = T.tanh(T.dot(self.Whh, h_t_minus_1) + T.dot(self.Whe, embedding) + self.Wb)
        # return next hidden state
        return [h_t, h_t]

    def final_state_given(self, h0, go_backwards=False):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     go_backwards=go_backwards,
                                     sequences=[self.sequence_embeddings],
                                     outputs_info=[h0, None])
        return h_t[-1]

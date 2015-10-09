import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class SimpleRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden, idxs=None, sequence_embeddings=None):
        assert (idxs is None) ^ (sequence_embeddings is None)
        if idxs is not None:
            # not tying weights, build our own set of embeddings
            self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx', orthogonal_init=True)
            self.sequence_embeddings = self.Wx[idxs]
            self.using_shared_embeddings = False
        else:
            # using tied weights, we won't be handling the update
            self.sequence_embeddings = sequence_embeddings
            self.using_shared_embeddings = True
        self.Whh = util.sharedMatrix(n_hidden, n_hidden, 'Whh', orthogonal_init=True)
        self.Whe = util.sharedMatrix(n_hidden, n_embedding, 'Whe', orthogonal_init=True)
        self.Wb = util.shared(util.zeros((n_hidden,)), 'Wb')

    def params_for_l2_penalty(self):
        params = [self.Whh, self.Whe, self.Wb]
        if not self.using_shared_embeddings:
            params.append(self.sequence_embeddings)
        return params

    def updates_wrt_cost(self, cost, learning_rate):
        # calculate dense updates
        params = [self.Whh, self.Whe, self.Wb]
        gradients = T.grad(cost=cost, wrt=params)
        updates = vanilla(params, gradients, learning_rate)
        # calculate a sparse update for embeddings if we are managing our own embedding matrix
        if not self.using_shared_embeddings:
            gradient = T.grad(cost=cost, wrt=self.sequence_embeddings)
            updates.append((self.Wx, T.inc_subtensor(self.sequence_embeddings, -learning_rate * gradient)))
        return updates

    def recurrent_step(self, embedding, h_t_minus_1):
        h_t = T.tanh(T.dot(self.Whh, h_t_minus_1) + T.dot(self.Whe, embedding) + self.Wb)
        return [h_t, h_t]

    def final_state_given(self, h0):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[self.sequence_embeddings],
                                     outputs_info=[h0, None])
        return h_t[-1]

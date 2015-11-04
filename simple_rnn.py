import numpy as np
import theano
import theano.tensor as T
from updates import vanilla, rmsprop
import util

class SimpleRnn(object):
    def __init__(self, name, n_in, n_embedding, n_hidden, opts, update_fn,
                 h0, idxs=None, sequence_embeddings=None):
        assert (idxs is None) ^ (sequence_embeddings is None)
        self.name_ = name
        self.update_fn = update_fn
        self.h0 = h0

        if idxs is not None:
            # not tying weights, build our own set of embeddings
            self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx', orthogonal_init=True)
            self.sequence_embeddings = self.Wx[idxs]
            self.using_shared_embeddings = False
        else:
            # using tied weights, we won't be handling the update
            self.sequence_embeddings = sequence_embeddings
            self.using_shared_embeddings = True

        # hidden -> hidden
        self.Whh = util.sharedMatrix(n_hidden, n_hidden, 'Whh', orthogonal_init=True)

        # embedded input -> hidden
        self.Whe = util.sharedMatrix(n_hidden, n_embedding, 'Whe', orthogonal_init=True)

        # bias
        self.Wb = util.shared(util.zeros((n_hidden,)), 'Wb')

    def name(self):
        return self.name_

    def dense_params(self):
        return [self.Whh, self.Whe, self.Wb]

    def params_for_l2_penalty(self):
        params = self.dense_params()
        if not self.using_shared_embeddings:
            params.append(self.sequence_embeddings)
        return params

    def updates_wrt_cost(self, cost, learning_rate):
        # calculate dense updates
        gradients = util.clipped(T.grad(cost=cost, wrt=self.dense_params()))
        updates = self.update_fn(self.dense_params(), gradients, learning_rate)
        # calculate a sparse update for embeddings if we are managing our own
        # embedding matrix
        if not self.using_shared_embeddings:
            gradient = util.clipped(T.grad(cost=cost, wrt=self.sequence_embeddings))
            updates.append((self.Wx, T.inc_subtensor(self.sequence_embeddings,
                                                     -learning_rate * gradient)))
        return updates

    def recurrent_step(self, embedding, h_t_minus_1):
        h_t = T.tanh(T.dot(self.Whh, h_t_minus_1) +
                     T.dot(self.Whe, embedding) +
                     self.Wb)
        return [h_t, h_t]

    def all_states(self):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[self.sequence_embeddings],
                                     outputs_info=[self.h0, None])
        return h_t

    def final_state(self):
        return self.all_states()[-1]

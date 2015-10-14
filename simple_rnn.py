import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class SimpleRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden, opts, update_fn,
                 idxs=None, sequence_embeddings=None, context=None):
        assert (idxs is None) ^ (sequence_embeddings is None)
        self.update_fn = update_fn

        # context represents (potential) extra input to each recurrent call for
        # nn_baseline this is not used but for seq2seq this is passed to s2 as the
        # output of s1
        self.context = context

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

        # we only need context -> hidden when we have a context that we need to mix
        # into our hidden state
        if self.context:
            # context -> hidden
            self.Whc = util.sharedMatrix(n_hidden, 2 * n_hidden, 'Whc',
                                         orthogonal_init=True)

    def dense_params(self):
        p = [self.Whh, self.Whe, self.Wb]
        if self.context:
            p.append(self.Whc)
        return p

    def params_for_l2_penalty(self):
        params = self.dense_params()
        if not self.using_shared_embeddings:
            params.append(self.sequence_embeddings)
        return params

    def updates_wrt_cost(self, cost, learning_rate):
        # calculate dense updates
        gradients = T.grad(cost=cost, wrt=self.dense_params())
        updates = self.update_fn(self.dense_params(), gradients, learning_rate)
        # calculate a sparse update for embeddings if we are managing our own
        # embedding matrix
        if not self.using_shared_embeddings:
            gradient = T.grad(cost=cost, wrt=self.sequence_embeddings)
            updates.append((self.Wx, T.inc_subtensor(self.sequence_embeddings,
                                                     -learning_rate * gradient)))
        return updates

    def recurrent_step(self, embedding, h_t_minus_1, context=None):
        hidden_state = T.dot(self.Whh, h_t_minus_1) + T.dot(self.Whe, embedding)
        if context:
            hidden_state += T.dot(self.Whc, context)
        hidden_state += self.Wb
        h_t = T.tanh(hidden_state)
        return [h_t, h_t]

    def final_state_given(self, h0):
        if self.context:
            [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                         sequences=[self.sequence_embeddings],
                                         non_sequences=[self.context],
                                         outputs_info=[h0, None])
        else:
            [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                         sequences=[self.sequence_embeddings],
                                         outputs_info=[h0, None])
        return h_t[-1]

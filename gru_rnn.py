import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class GruRnn(object):
    def __init__(self, name, n_in, n_embedding, n_hidden, opts, update_fn,
                 idxs=None, sequence_embeddings=None, context=None):
        assert (idxs is None) ^ (sequence_embeddings is None)
        self.name_ = name
        self.update_fn = update_fn

        if idxs is not None:
            # not tying weights, build our own set of embeddings
            self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx', orthogonal_init=True)
            self.sequence_embeddings = self.Wx[idxs]
            self.using_shared_embeddings = False
        else:
            # using tied weights, we won't be handling the update
            self.sequence_embeddings = sequence_embeddings
            self.using_shared_embeddings = True

        # params for standard recurrent step
        self.Uh = util.sharedMatrix(n_hidden, n_hidden, 'Uh', orthogonal_init=True)
        self.Wh = util.sharedMatrix(n_hidden, n_embedding, 'Wh', orthogonal_init=True)
        self.bh = util.shared(util.zeros((n_hidden,)), 'bh')

        # params for reset gate; initial bias to not reset
        self.Ur = util.sharedMatrix(n_hidden, n_hidden, 'Ur', orthogonal_init=True)
        self.Wr = util.sharedMatrix(n_hidden, n_embedding, 'Wr', orthogonal_init=True)
        self.br = util.shared(np.asarray([opts.gru_initial_bias]*n_hidden), 'br')

        # params for carry gate; initial bias to never carry h_t_minus_1
        self.Uz = util.sharedMatrix(n_hidden, n_hidden, 'Uz', orthogonal_init=True)
        self.Wz = util.sharedMatrix(n_hidden, n_embedding, 'Wz', orthogonal_init=True)
        self.bz = util.shared(np.asarray([opts.gru_initial_bias]*n_hidden), 'bz')

    def name(self):
        return self.name_

    def dense_params(self):
        return [self.Uh, self.Wh, self.bh, 
                self.Ur, self.Wr, self.br, 
                self.Uz, self.Wz, self.bz]

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

    def recurrent_step(self, embedding, h_t_minus_1):
        # reset gate; how much will we zero out h_t_minus_1 in our candidate
        # next hidden state calculation?
        r = T.nnet.sigmoid(T.dot(self.Ur, h_t_minus_1) + T.dot(self.Wr, embedding) +
                           self.br)
        # candidate hidden state
        h_t_candidate = T.tanh(r * T.dot(self.Uh, h_t_minus_1) +
                               T.dot(self.Wh, embedding) + self.bh)
        # carry gate; how much of h_t_minus_1 will we take with h_candidate?
        z = T.nnet.sigmoid(T.dot(self.Uz, h_t_minus_1) + T.dot(self.Wz, embedding) +
                           self.bz)
        # actual hidden state affine combo of last state and candidate state
        h_t = (1 - z) * h_t_minus_1 + z * h_t_candidate
        return [h_t, h_t]

    def final_state_given(self, h0):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[self.sequence_embeddings],
                                     outputs_info=[h0, None])
        return h_t[-1]

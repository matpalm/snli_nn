import numpy as np
import util
import theano
import theano.tensor as T

class SimpleRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden, orthogonal_init):
        self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx', orthogonal_init)
        self.Wrec = util.sharedMatrix(n_hidden, n_embedding, 'Wrec', orthogonal_init)

    def params(self):
        return [self.Wx, self.Wrec] #, self.Wy]

    def recurrent_step(self, x_t, h_t_minus_1):
        # calc new hidden state; elementwise add of embedded input &
        # recurrent weights dot _last_ hiddenstate
        embedding = self.Wx[x_t]
        h_t = T.tanh(h_t_minus_1 + T.dot(self.Wrec, embedding))
        # return next hidden state
        return [h_t, h_t]

    def final_state_given(self, x, h0):
        [_h_t, h_t], _ = theano.scan(fn=self.recurrent_step,
                                     sequences=[x],
                                     outputs_info=[h0, None])
        return h_t[-1]

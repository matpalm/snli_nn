from gru_rnn import *
import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class BidirectionalGruRnn(object):
    def __init__(self, name, n_in, n_embedding, n_hidden, opts, update_fn, h0, idxs):
        self.name_ = name
        # TODO: support tied embeddings again
        self.forward_gru = GruRnn(("f_%s" % name), n_in, n_embedding, n_hidden, opts, 
                                  update_fn, h0, idxs=idxs)
        self.backwards_gru = GruRnn(("f_%s" % name), n_in, n_embedding, n_hidden, opts, 
                                    update_fn, h0, idxs=idxs[::-1])
    
    def name(self):
        return self.name_

# DONT need this (?) only used for updates_wrt_cost per non abstract?
#    def dense_params(self):
#        return self.forward_gru.dense_params() + self.backwards_gru.dense_params()

    def params_for_l2_penalty(self):
        return self.forward_gru.params_for_l2_penalty() + \
            self.backwards_gru.params_for_l2_penalty()

    def updates_wrt_cost(self, cost, learning_rate):
        return self.forward_gru.updates_wrt_cost(cost, learning_rate) + \
            self.backwards_gru.updates_wrt_cost(cost, learning_rate)

    # return hidden activations for recurrent step (as a 2-tuple)
    # [(f_s1, b_sn), (f_s2, b_sn-1), ...]
    def all_hidden_states(self):
        forwards_ht = self.forward_gru.all_hidden_states()
        backwards_ht = self.backwards_gru.all_hidden_states()
        return zip(forwards_ht, backwards_ht[::-1])

    # [final forward state, final backwards state]
    def final_states(self):
        return [self.forward_gru.final_state(), self.backwards_gru.final_state()]


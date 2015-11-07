from embeddings import Embeddings
from gru_rnn import *
import numpy as np
import util
import theano
import theano.tensor as T
from updates import vanilla, rmsprop

class BidirectionalGruRnn(object):
    def __init__(self, name, vocab_size, embedding_dim, hidden_dim, opts, update_fn, h0,
                 idxs):
        self.name_ = name

        def build_gru(name, idxs):
            embeddings = Embeddings(vocab_size, embedding_dim, idxs=idxs)
            return GruRnn(name, embedding_dim, hidden_dim, opts, update_fn, h0,
                          embeddings.embeddings())

        # TODO: support tied embeddings again
        self.forward_gru = build_gru(name=("f_%s" % name), idxs=idxs)
        self.backwards_gru = build_gru(name=("b_%s" % name), idxs=idxs[::-1])
    
    def name(self):
        return self.name_

# DONT need this (?) only used for updates_wrt_cost per non abstract?
#    def dense_params(self):
#        return self.forward_gru.dense_params() + self.backwards_gru.dense_params()

    def params_for_l2_penalty(self):
        return self.forward_gru.params_for_l2_penalty() + \
            self.backwards_gru.params_for_l2_penalty()

    def updates_wrt_cost(self, cost, learning_rate):
        print "BIDIR UPDATES"
        return self.forward_gru.updates_wrt_cost(cost, learning_rate) + \
            self.backwards_gru.updates_wrt_cost(cost, learning_rate)

    # return hidden activations for recurrent step (as a 2-tuple)
    # [f_s1 ++ b_sn, f_s2 ++ b_sn-1, ...]
    def all_states(self):
        forwards_ht = self.forward_gru.all_states()
        backwards_ht = self.backwards_gru.all_states()
        all_states, _ = theano.scan(fn=lambda f, b: T.concatenate([f, b]),
                                    sequences=[forwards_ht, backwards_ht],
                                    outputs_info=[None])
        return all_states

    # [final forward state, final backwards state]
    def final_states(self):
        return T.concatenate([self.forward_gru.final_state(),
                              self.backwards_gru.final_state()])


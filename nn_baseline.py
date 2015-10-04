#!/usr/bin/env python
import argparse
from concat_with_softmax import ConcatWithSoftmax
import json
import numpy as np
from simple_rnn import SimpleRnn
from sklearn.metrics import confusion_matrix
import sys
import time
import theano
import theano.tensor as T
import util
from updates import vanilla, rmsprop
from vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument("--train-set", default="data/snli_1.0_train.jsonl")
parser.add_argument("--num-from-train", default=-1, type=int)
parser.add_argument("--dev-set", default="data/snli_1.0_dev.jsonl")
parser.add_argument("--num-from-dev", default=-1, type=int)
parser.add_argument("--dev-run-freq", default=-1, type=int)
parser.add_argument("--num-epochs", default=-1, type=int)
parser.add_argument('--learning-rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--adaptive-learning-rate-fn', default='vanilla', help='vanilla (sgd) or rmsprop')
parser.add_argument('--embedding-dim', default=3, type=int, help='embedding node dimensionality')
parser.add_argument('--hidden-dim', default=4, type=int, help='hidden node dimensionality')
opts = parser.parse_args()
print >>sys.stderr, opts

NUM_LABELS = 3

# slurp training data, including converting of tokens -> ids
vocab = Vocab()
train_x, train_y, train_stats = util.load_data(opts.train_set, vocab,
                                               update_vocab=True,
                                               max_egs=int(opts.num_from_train))
print >>sys.stderr, "train_stats", len(train_x), train_stats
dev_x, dev_y, dev_stats = util.load_data(opts.dev_set, vocab,
                                         update_vocab=False,
                                         max_egs=int(opts.num_from_dev))
print >>sys.stderr, "dev_stats", len(dev_x), dev_stats

# input/output vars
s1_idxs = T.ivector('s1')  # sequence for sentence one
s2_idxs = T.ivector('s2')  # sequence for sentence two
actual_y = T.ivector('y')  # single for sentence pair label; 0, 1 or 2

# shared initial zero hidden state
h0 = theano.shared(np.zeros(opts.hidden_dim, dtype='float32'), name='h0', borrow=True)

# build rnn for pass over s1
config = (vocab.size(), opts.embedding_dim, opts.hidden_dim, True)
s1_rnn = SimpleRnn(*config)
final_s1_state = s1_rnn.final_state_given(s1_idxs, h0)

# build another rnn for pass over s2
s2_rnn = SimpleRnn(*config)
final_s2_state = s2_rnn.final_state_given(s2_idxs, h0)

# concat, do a final linear combo and apply softmax
concat_with_softmax = ConcatWithSoftmax([final_s1_state, final_s2_state], NUM_LABELS, opts.hidden_dim)
prob_y, pred_y = concat_with_softmax.prob_pred()

# calc xent and get each layer to provide updates
cross_entropy = T.mean(T.nnet.categorical_crossentropy(prob_y, actual_y))
updates = s1_rnn.updates_wrt_cost(cross_entropy, opts.learning_rate) + \
          s2_rnn.updates_wrt_cost(cross_entropy, opts.learning_rate) + \
          concat_with_softmax.updates_wrt_cost(cross_entropy, opts.learning_rate)

print >>sys.stderr, "compiling"
train_fn = theano.function(inputs=[s1_idxs, s2_idxs, actual_y],
                           outputs=[],
                           updates=updates)
test_fn = theano.function(inputs=[s1_idxs, s2_idxs],
                          outputs=[pred_y])

def test_on_dev_set():
    actuals = []
    predicteds  = []
    for n, ((s1, s2), y) in enumerate(zip(dev_x, dev_y)):
        pred_y, = test_fn(s1, s2)
        actuals.append(y)
        predicteds.append(pred_y)
    dev_c = confusion_matrix(actuals, predicteds)
    dev_c_accuracy = util.accuracy(dev_c)
    print "dev confusion\n %s (%s)" % (dev_c, dev_c_accuracy)
    s1_wb_norm = np.linalg.norm(s1_rnn.Wb.get_value())
    s2_wb_norm = np.linalg.norm(s2_rnn.Wb.get_value())
    print "s1.Wb", s1_rnn.Wb.get_value()
    print "s2.Wb", s2_rnn.Wb.get_value()
    print "STATS\t%s" % "\t".join(map(str, [dev_c_accuracy, s1_wb_norm, s2_wb_norm]))
    sys.stdout.flush()

print >>sys.stderr, "training"
epoch = 0
n_egs_since_dev_test = 0
while epoch != opts.num_epochs:
    print ">epoch %s (%s)" % (epoch, time.strftime("%Y-%m-%d %H:%M:%S"))
    for (s1, s2), y in zip(train_x, train_y):
        train_fn(s1, s2, [y])
        n_egs_since_dev_test += 1
        if n_egs_since_dev_test == opts.dev_run_freq:
            test_on_dev_set()
            n_egs_since_dev_test = 0
    epoch += 1

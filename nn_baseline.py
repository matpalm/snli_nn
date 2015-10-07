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
parser.add_argument("--max-run-time-sec", default=-1, type=int)
parser.add_argument('--learning-rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--adaptive-learning-rate-fn', default='vanilla', help='vanilla (sgd) or rmsprop')
parser.add_argument('--embedding-dim', default=3, type=int, help='embedding node dimensionality')
parser.add_argument('--hidden-dim', default=4, type=int, help='hidden node dimensionality')
parser.add_argument('--bidirectional', action='store_true', help='whether to build bidirectional rnns for s1 & s2')
opts = parser.parse_args()
print >>sys.stderr, opts

NUM_LABELS = 3

def dts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print >>sys.stderr, dts(), s

# slurp training data, including converting of tokens -> ids
vocab = Vocab()
train_x, train_y, train_stats = util.load_data(opts.train_set, vocab,
                                               update_vocab=True,
                                               max_egs=int(opts.num_from_train))
log("train_stats %s %s" % (len(train_x), train_stats))
dev_x, dev_y, dev_stats = util.load_data(opts.dev_set, vocab,
                                         update_vocab=False,
                                         max_egs=int(opts.num_from_dev))
log("dev_stats %s %s" % (len(dev_x), dev_stats))

# input/output vars
s1_idxs = T.ivector('s1')  # sequence for sentence one
s2_idxs = T.ivector('s2')  # sequence for sentence two
actual_y = T.ivector('y')  # single for sentence pair label; 0, 1 or 2

# shared initial zero hidden state
h0 = theano.shared(np.zeros(opts.hidden_dim, dtype='float32'), name='h0', borrow=True)

# build seperate rnns for passes over s1/s2 with optional bidirectional passes.
def rnn(idxs, forwards):
    return SimpleRnn(vocab.size(), opts.embedding_dim, opts.hidden_dim, True, idxs, forwards)

rnns = [rnn(s1_idxs, forwards=True), rnn(s2_idxs, forwards=True)]
if opts.bidirectional:
    rnns += [rnn(s1_idxs, forwards=False), rnn(s2_idxs, forwards=False)]
final_rnn_states = [rnn.final_state_given(h0) for rnn in rnns]

# concat final states of rnns, do a final linear combo and apply softmax for prediction.
concat_with_softmax = ConcatWithSoftmax(final_rnn_states, NUM_LABELS, opts.hidden_dim)
prob_y, pred_y = concat_with_softmax.prob_pred()

# calc xent and get each layer to provide updates
cross_entropy = T.mean(T.nnet.categorical_crossentropy(prob_y, actual_y))
layers = rnns + [concat_with_softmax]
updates = []
for layer in layers:
    updates.extend(layer.updates_wrt_cost(cross_entropy, opts.learning_rate))

log("compiling")
train_fn = theano.function(inputs=[s1_idxs, s2_idxs, actual_y],
                           outputs=[cross_entropy],
                           updates=updates)
test_fn = theano.function(inputs=[s1_idxs, s2_idxs, actual_y],
                          outputs=[pred_y, cross_entropy])

def test_on_dev_set():
    actuals = []
    predicteds  = []
    costs = []
    for (s1, s2), y in zip(dev_x, dev_y):
        pred_y, cost = test_fn(s1, s2, [y])
        actuals.append(y)
        predicteds.append(pred_y)
        costs.append(cost)
    dev_c = confusion_matrix(actuals, predicteds)
    dev_c_accuracy = util.accuracy(dev_c)
    print "dev confusion\n %s (%s)" % (dev_c, dev_c_accuracy)
    return dev_c_accuracy, {"mean": float(np.mean(costs)), "sd": float(np.std(costs))}

START_TIME = int(time.time())
def stats(d):
    d['dts_h'] = dts()
    d['elapsed_time'] = int(time.time()) - START_TIME
    print "STATS\t%s" % json.dumps(d)
    sys.stdout.flush()

log("training")
epoch = 0
n_egs_trained = 0
training_early_stop_time = opts.max_run_time_sec + time.time()
run = "RUN_%s" % START_TIME
while epoch != opts.num_epochs:
    costs = []
    for (s1, s2), y in zip(train_x, train_y):
        cost, = train_fn(s1, s2, [y])
        costs.append(cost)
        n_egs_trained += 1
        early_stop = False
        if opts.max_run_time_sec != -1 and time.time() > training_early_stop_time:
            early_stop = True
        if n_egs_trained % opts.dev_run_freq == 0 or early_stop:
            dev_accuracy, dev_cost_stats = test_on_dev_set()
            stats({"run": run, "epoch": epoch, "n_egs_trained": n_egs_trained,
                   "e_dim": opts.embedding_dim, "h_dim": opts.hidden_dim,
                   "lr": opts.learning_rate, "dev_acc": dev_accuracy,
                   "train_cost": {"mean": float(np.mean(costs)), "sd": float(np.std(costs))},
                   "dev_cost": dev_cost_stats,
                   "bidir": opts.bidirectional})
            costs = []
        if early_stop:
            exit(0)
    epoch += 1

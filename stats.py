import json
import numpy as np
import os
import sys
import time
import util

class Stats(object):
    def __init__(self, opts):
        self.start_time = int(time.time())
        self.n_egs_trained = 0
        self.base_stats = {"run": "RUN_%s_%s" % (self.start_time, os.getpid()),
                           "e_dim": opts.embedding_dim, "h_dim": opts.hidden_dim,
                           "lr": opts.learning_rate, "l2_penalty": opts.l2_penalty,
                           "bidir": opts.bidirectional, "tied_embeddings": opts.tied_embeddings}
        self.reset()

    def record_training_cost(self, cost):
        self.train_costs.append(cost)
        self.n_egs_trained += 1

    def record_dev_cost(self, cost):
        self.dev_costs.append(cost)

    def set_dev_accuracy(self, dev_accuracy):
        assert self.dev_accuracy is None
        self.dev_accuracy = dev_accuracy

    def flush_to_stdout(self, epoch):
        stats = dict(self.base_stats)
        stats.update({"dts_h": util.dts(), "epoch": epoch,
                      "n_egs_trained": self.n_egs_trained,
                      "elapsed_time": int(time.time()) - self.start_time,
                      "train_cost": util.mean_sd(self.train_costs),
                      "dev_cost": util.mean_sd(self.dev_costs),
                      "dev_acc": self.dev_accuracy})
        print "STATS\t%s" % json.dumps(stats)
        sys.stdout.flush()
        self.reset()

    def reset(self):
        self.train_costs = []
        self.dev_costs = []
        self.dev_accuracy = None


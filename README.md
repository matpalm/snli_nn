# snli hacking

hacking with the Stanford Natural Language Inference corpus http://nlp.stanford.edu/projects/snli/

# tldr results

( all for 100 dev )

model | # train | dev accuracy
----- | ----- | --------
log_reg_baseline.py | 100 | 0.35
log_reg_baseline.py | 10K | 0.56
log_reg_baseline.py | 500K+ (all) | 0.67
nn_baseline.py (elman) | 100 | 0.39
nn_baseline.py (elman) | 10K | 0.58
nn_baseline.py (elman) | 500K+ (all) | 0.63
nn_baseline.py (gru) | 500K+ (all) | 0.70 |

# simple logistic regression baseline

features...

* all tokens in sentence 1 prepended with "s1_"
* all tokens in sentence 2 prepended with "s2_"

trained on all data; tested on all dev

```
$ time ./log_reg_baseline.py

train confusion
 [[121306  27808  34073]
  [ 29941 117735  35088]
  [ 23662  20907 138847]] (accuracy 0.687)

dev confusion
 [[2077  549  652]
  [ 546 2044  645]
  [ 474  404 2451]] (accuracy 0.667)

# approx 6m
```

# nn models

three nn models

* nn_baseline: uni/bidirectional rnns (simple/gru) over s1/s2; concatenated states; MLP to softmax
* nn_seq2seq: bidirectional rnn over s1; first & last state concatenated; feed as context to bidirectional rnn over s2; MLP to softmax (WIP)
* nn_seq2seq_attention: as nn_seq2seq but attend back to all states of s1, not just first/last; MLP to softmax (WIP)

## nn_baseline

```
usage: nn_baseline.py [-h] [--train-set TRAIN_SET]
                      [--num-from-train NUM_FROM_TRAIN] [--dev-set DEV_SET]
                      [--num-from-dev NUM_FROM_DEV]
                      [--dev-run-freq DEV_RUN_FREQ] [--num-epochs NUM_EPOCHS]
                      [--max-run-time-sec MAX_RUN_TIME_SEC]
                      [--learning-rate LEARNING_RATE] [--update-fn UPDATE_FN]
                      [--embedding-dim EMBEDDING_DIM]
                      [--hidden-dim HIDDEN_DIM] [--bidirectional]
                      [--tied-embeddings] [--l2-penalty L2_PENALTY]
                      [--rnn-type RNN_TYPE]
                      [--gru-initial-bias GRU_INITIAL_BIAS]

optional arguments:
  -h, --help            show this help message and exit
  --train-set TRAIN_SET
  --num-from-train NUM_FROM_TRAIN
                        number of egs to read from train. -1 => all
  --dev-set DEV_SET
  --num-from-dev NUM_FROM_DEV
                        number of egs to read from dev. -1 => all
  --dev-run-freq DEV_RUN_FREQ
                        frequency (in num examples trained) to run against dev
                        set
  --num-epochs NUM_EPOCHS
                        number of epoches to run. -1 => forever
  --max-run-time-sec MAX_RUN_TIME_SEC
                        max secs to run before early stopping. -1 => dont
                        early stop
  --learning-rate LEARNING_RATE
                        learning rate
  --update-fn UPDATE_FN
                        vanilla (sgd) or rmsprop
  --embedding-dim EMBEDDING_DIM
                        embedding node dimensionality
  --hidden-dim HIDDEN_DIM
                        hidden node dimensionality
  --bidirectional       whether to build bidirectional rnns for s1 & s2
  --tied-embeddings     whether to tie embeddings for each RNN
  --l2-penalty L2_PENALTY
                        l2 penalty for params
  --rnn-type RNN_TYPE   Rnn cell type {SimpleRnn,GruRnn}
  --gru-initial-bias GRU_INITIAL_BIAS
                        initial bias for r & z for GruRnn. higher => more like
                        SimpleRnn
```

### first run unidirectional

* two rnns; one for each sentence
* concat output, one layer MLP and softmax over 3 classes

```
$ ./nn_baseline.py --embedding-dim=50 --hidden-dim=50 \
 --learning-rate=0.01 --dev-run-freq=10000
```

![baseline](imgs/baseline.png?raw=true "baseline")

(vertical line denotes epoch)

### with bidirectional

* another two rnns; in opposite directions
* all 4 outputs concatted before MLP & softmax

### checking learning rates

```
export COMMON="--embedding-dim=50 --hidden-dim=50 --dev-run-freq=100000 --bidirectional"
./nn_baseline.py $COMMON --learning-rate=0.1
./nn_baseline.py $COMMON --learning-rate=0.01
./nn_baseline.py $COMMON --learning-rate=0.001
```

![lr_comparison](imgs/lr_comparison.png?raw=true "lr_comparison")

so not stable at 0.1. still running sgd; 

### with tied embeddings & l2 penalty

```
export COMMON="--embedding-dim=50 --hidden-dim=50 --learning-rate=0.01 --dev-run-freq=100000"
./nn_baseline.py $COMMON
./nn_baseline.py $COMMON --tied-embeddings
./nn_baseline.py $COMMON --bidirectional
./nn_baseline.py $COMMON --bidirectional --tied-embeddings
```

![tied_comparison_dev_acc](imgs/tied_comparison_dev_acc.png?raw=true "tied_comparison dev accuracy")

### gru vs simple

up until now everything was a simple elman network, let's try a [gru](http://arxiv.org/abs/1412.3555)

```
export C="--learning-rate=0.01 --dev-run-freq=10000 --bidirectional --tied-embeddings --embedding-dim=100 --hidden-dim=100"
./nn_baseline.py $C --rnn-type=SimpleRnn
./nn_baseline.py $C --rnn-type=GruRnn
```

![simple_vs_gru](imgs/simple_vs_gru.png?raw=true "simple vs gru dev accuracy")

### TODOS

* neutral examples are non symmetric, should swap them 0.5 during training
* preloading of data; it's slow to start
* unrolling? maybe not bother for hacking. might be finally up to a point where batching speed matters...
* unidir on s2 attending back to bidir run over s1; then just MLP on s2 output

## nn_seq2seq

* bidir on s1; keep start / last concatted state
* unidir on s2 with added context (per timestep) directly from s1 output

## nn_seq2seq_attention

* bidir on s1; keep all states concatted
* unidir on s2 with input attended over s1 states

# appendix: vocab check

```
time cat data/snli_1.0_train.jsonl \
 | ./parse_distinct_tokens.py \
 | sort -k2 -nr \
 > token_freq.tsv
```

36_391 entries (nice and small!)

but an unusual set compared to, say, the 1e6 sentence corpus...

token | 1e6 freq | token | snli freq
----- | -------- | ----- | ---------
. | 970059 | a | 1441039
of | 845203 | . | 964030
and | 645219 | the | 535493
in | 602064 | in | 407662
to | 488035 | is | 374068
a | 482654 | man | 266490
is | 243535 | on | 236203
' | 241019 | and | 206529
was | 239712 | are | 199381
-lrb- | 237887 | of | 192415
-rrb- | 237400 | with | 169513
` | 212268 | woman | 137794
as | 197400 | two | 122247
for | 185954 | people | 121335
by | 162542 | , | 114538
with | 162149 | to | 114072
on | 160348 | at | 98790
that | 150584 | wearing | 81141
's | 148546 | an | 80334
'' | 124864 | his | 72550
`` | 122000 | young | 61596
from | 110950 | men | 61112
his | 109739 | playing | 59568
he | 109146 | girl | 59345
it | 108952 | boy | 58354
at | 100304 | white | 57115
are | 93788 | shirt | 56578
an | 87625 | while | 56323
were | 85952 | black | 55133
which | 83635 | dog | 54026

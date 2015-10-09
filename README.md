# snli hacking

hacking with the Stanford Natural Language Inference corpus http://nlp.stanford.edu/projects/snli/

# tldr results

( all for 100 dev )

model | # train | dev accuracy
----- | ----- | --------
log_reg_baseline.py | 100 | 0.35
log_reg_baseline.py | 1K | 0.45
log_reg_baseline.py | 10K | 0.56
log_reg_baseline.py | 500K+ (all) | 0.67
nn_baseline.py | 100 | 0.39
nn_baseline.py | 1K | 0.47
nn_baseline.py | 10K | 0.58
nn_baseline.py | 500K_ (all) | 0.63

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
  [ 23662  20907 138847]] (0.687860756107)

dev confusion
 [[2077  549  652]
  [ 546 2044  645]
  [ 474  404 2451]] (0.667750457224)

# approx 6m
```

# nn models

## baseline

* two rnns; one for each sentence
* concat output, one layer MLP and softmax over 3 classes

```
$ ./nn_baseline.py --embedding-dim=50 --hidden-dim=50 --learning-rate=0.01 --dev-run-freq=10000 | tee runs/baseline.50.50
```

![baseline](imgs/baseline.png?raw=true "baseline")

(vertical line denotes epoch)

## with bidirectional

* another two rnns; in opposite directions
* all 4 outputs concatted before MLP & softmax

## checking learning rates

```
export COMMON="--embedding-dim=50 --hidden-dim=50 --dev-run-freq=100000 --bidirectional"
./nn_baseline.py $COMMON --learning-rate=0.1
./nn_baseline.py $COMMON --learning-rate=0.01
./nn_baseline.py $COMMON --learning-rate=0.001
```

![lr_comparison](imgs/lr_comparison.png?raw=true "lr_comparison")

so not stable at 0.1. still running sgd; 

## with tied weights & l2 penalty

```
export COMMON="--embedding-dim=50 --hidden-dim=50 --learning-rate=0.01 --dev-run-freq=100000"
./nn_baseline.py $COMMON | tee runs/onight.20151009.notied.nobidir
./nn_baseline.py $COMMON --tied-embeddings | tee runs/onight.20151009.tied.nobidir
./nn_baseline.py $COMMON --bidirectional | tee runs/onight.20151009.notied.bidir
./nn_baseline.py $COMMON --bidirectional --tied-embeddings | tee runs/onight.20151009.tied.bidir
```

![tied_comparison_dev_acc](imgs/tied_comparison_dev_acc.png?raw=true "tied_comparison dev accuracy")

# TODOS

* rmsprop for non embeddings; or at least some learning rate management.
* preloading of data; it's slow to start
* grus
* neutral examples are non symmetric, should swap them 0.5 during training
* unrolling? maybe not bother for hacking. might be finally up to a point where batching speed matters...
* unidir on s2 attending back to bidir run over s1; then just MLP on s2 output

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

import numpy as np
import theano
from theano.ifelse import ifelse
import theano.tensor as T

RND = np.random.RandomState()
RND_STREAM = T.shared_randomstreams.RandomStreams(RND.randint(1000000000))

APPLY_DROPOUT = 1
NO_DROPOUT = 0

# tensor: tensor to apply dropout to
# train: bscalar; either APPLY_DROPOUT or NO_DROPOUT depending on train/test time
# keep_prob: a fscalar; 0.0<v<1.0 denoting keep value probability. 
#            value 1.0 => no dropout, use this for inference.
def dropout(tensor, apply_dropout, keep_prob):
    mask = RND_STREAM.binomial(n=1, p=keep_prob, size=tensor.shape, dtype='float32')
    keep_prob = T.cast(keep_prob, 'float32')  # todo: weirdity around shared.set_value
    tensor_dropped = tensor * (1.0 / keep_prob) * mask
    return ifelse(apply_dropout, tensor_dropped, tensor)

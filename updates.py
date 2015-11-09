import numpy as np
import theano
import theano.tensor as T
import util

def vanilla(params, gradients, opts):
    return [(param, param - opts.learning_rate * gradient) 
            for param, gradient in zip(params, gradients)]

def momentum(params, gradients, opts):
    assert opts.momentum >= 0.0 and opts.momentum <= 1.0
    updates = []
    for param, gradient in zip(params, gradients):
        velocity_t0 = util.zeros_in_the_shape_of(param)
        velocity_t1 = opts.momentum * velocity_t0 - opts.learning_rate * gradient
        updates.append((velocity_t0, velocity_t1))
        updates.append((param, param + velocity_t1))
    return updates

def rmsprop(params, gradients, opts):
    assert opts.momentum
    assert opts.momentum >= 0.0 and opts.momentum <= 1.0
    updates = []
    for param_t0, gradient in zip(params, gradients):
        # rmsprop see slide 29 of http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        # first the mean_sqr exponential moving average
        mean_sqr_t0 = util.zeros_in_the_shape_of(param_t0)
        mean_sqr_t1 = (opts.momentum * mean_sqr_t0) + ((1.0-opts.momentum) * gradient**2)
        updates.append((mean_sqr_t0, mean_sqr_t1))
        # update param surpressing gradient by this average
        param_t1 = param_t0 - opts.learning_rate * (gradient / T.sqrt(mean_sqr_t1 + 1e-10))
        updates.append((param_t0, param_t1))
    return updates

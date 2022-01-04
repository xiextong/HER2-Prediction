from keras import backend as K
from all_index_bin import *
import tensorflow as tf
############################################################
import numpy as np
from functools import partial, update_wrapper

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

_EPSILON = 1e-7

def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON

def binary_crossentropy_weighted(target, output, weight):

    _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    return tf.nn.weighted_cross_entropy_with_logits(targets=target,logits=output,pos_weight=weight)

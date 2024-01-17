# Copyright (C) 2017 Hewlett Packard Enterprise Development LP
# Modifications copyright (C) 2023 Computing Systems Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from numbers import Number
from typing import Any, List, Optional

import numpy as np
from torch import Tensor
from torch import nn
from torch import finfo, iinfo, is_floating_point
from torch import dtype as tdt

# From https://github.com/sergey-serebryakov/nns/blob/master/nns/nns.py


def _get_dims(val: Any) -> Optional[List[int]]:
    """
    Get the dims from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None


def _gru_op_flops_handler(inputs: List[Any],
                            outputs: List[Any]) -> Number:
    return _gru_op_handler(inputs, outputs, 'FLOPs')


def _gru_op_acts_handler(inputs: List[Any],
                            outputs: List[Any]) -> Number:
    return _gru_op_handler(inputs, outputs, 'activations')


def _gru_op_handler(inputs: List[Any],
                        outputs: List[Any],
                        kind: str='FLOPs') -> Number:

    input_dims = [_get_dims(v) for v in inputs]
    output_dims = [_get_dims(v) for v in outputs]

    # print(input_dims)
    # print(output_dims)

    seq_len = input_dims[0][1]
    input_size = input_dims[0][-1]
    output_size = output_dims[0][-1]

    num_layers = output_dims[1][0]

    if kind == 'FLOPs':

        flops_total = 0

        for layer in range(num_layers):

            flops_r_t = seq_len*output_size*(input_size + output_size)
            flops_z_t = flops_r_t
            flops_n_t = flops_r_t + seq_len*output_size
            flops_h_t = 4*seq_len*output_size

            flops_total += flops_r_t + flops_z_t + flops_n_t + flops_h_t

            input_size = output_size

        return flops_total

    elif kind == 'activations':
        # Assumed implementation (all output shapes below: [output_size,]):
        #    1.  r_ir = W_ir*x_t + b_ir                                1
        #    2.  r_hr = W_hr*h_(t-1) + b_hr                            1
        #    3.  r_t = sigmoid(x_r + r_r)                              2
        #    4.  z_iz = W_iz*x_t + b_iz                                1
        #    5.  z_hz = W_hz*h_(t-1) + b_hz                            1
        #    6.  z_t = sigmoid(x_z + r_z)                              2
        #    7.  n_in = W_in*x_t + b_in                                1
        #    8.  n_hn = W_hn*h_(t.1) + b_hn                            1
        #    9.  n_r_t = n_hn.*r_t                                     1
        #    10. n_t = tanh(n_hn + n_r_t)                              1
        #    11. h_zn_t = (1 - z_t).*n_(t-1)                           2
        #    12. h_zh_t = z_t.*h_t                                     1
        #    13. h_t = h_zn_t + h_zh_t                                 1
        #
        #    Total activations:                                       16

        return 16*num_layers*seq_len*output_size

    else:
        raise NotImplementedError(f'Metric {kind} not implemented')
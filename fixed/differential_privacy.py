# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for differential privacy."""


from logging import WARNING, INFO
from typing import Optional

import numpy as np

from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log

# to measure the size (magnitude) of an update/parameter vector.
def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
    # pylint: disable=consider-using-generator
    return float(np.sqrt(sum([norm**2 for norm in array_norms])))

# Adds i.i.d. Gaussian noise N(0, std_dev^2) to every entry of every tensor, in place.
def add_gaussian_noise_inplace(input_arrays: NDArrays, std_dev: float) -> None:
    """Add Gaussian noise to each element of the input arrays."""
    print("Before noise is : {0}".format(input_arrays[0][0][0]))
    for array in input_arrays:
        array += np.random.normal(0, std_dev, array.shape)
    print("After noise is : {0}".format(input_arrays[0][0][0]))

# Computes the L2 norm of the whole vector (all tensors combined)        
def clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> None:
    """Clip model update based on the clipping norm in-place.

    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    # scaling_factor = min(1, clipping_norm / input_norm) - Changed to avoid division by zero
    if input_norm == 0:
        scaling_factor = 1.0  # No scaling needed
    else:
        scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor

# for server-side/aggregated updates - Not LDP
def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)

#Disabled by Kasra
# replace param1 with a clipped step away from param2
def compute_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> None:
    """Compute model update (param1 - param2) and clip it.

    Then add the clipped value to param1."""
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]
        # param1[i] = param2[i]

# Like clip_inputs_inplace, but returns True if it actually had to shrink (scaling factor < 1)
def adaptive_clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> bool:
    """Clip model update based on the clipping norm in-place.

    It returns true if scaling_factor < 1 which is used for norm_bit
    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor
    return scaling_factor < 1

# Same idea as compute_clip_model_update, but
# Using the adaptive variant that tells you if clipping happened this round
def compute_adaptive_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> bool:
    """Compute model update, clip it, then add the clipped value to param1.

    model update = param1 - param2
    Return the norm_bit
    """
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    norm_bit = adaptive_clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]

    return norm_bit

# Central DP - not LDP
# converts Flower Parameters → arrays
# computes std using compute_stdv(...)
# adds Gaussian noise, converts back.
def add_gaussian_noise_to_params(
    model_params: Parameters,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
) -> Parameters:
    """Add gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
    )
    return ndarrays_to_parameters(model_params_ndarrays)

# Math helper for adaptive clipping 
# If you want a target effective noise_multiplier on the clipped-count statistic, 
# it tells you how much noise you actually need to add,
# and warns if it’s too much given your clipped_count_stddev.
def compute_adaptive_noise_params(
    noise_multiplier: float,
    num_sampled_clients: float,
    clipped_count_stddev: Optional[float],
) -> tuple[float, float]:
    """Compute noising parameters for the adaptive clipping.

    Paper: https://arxiv.org/abs/1905.03871
    """
    if noise_multiplier > 0:
        if clipped_count_stddev is None:
            clipped_count_stddev = num_sampled_clients / 20
        if noise_multiplier >= 2 * clipped_count_stddev:
            raise ValueError(
                f"If not specified, `clipped_count_stddev` is set to "
                f"`num_sampled_clients`/20 by default. This value "
                f"({num_sampled_clients / 20}) is too low to achieve the "
                f"desired effective `noise_multiplier` ({noise_multiplier}). "
                f"Consider increasing `clipped_count_stddev` or decreasing "
                f"`noise_multiplier`."
            )
        noise_multiplier_value = (
            noise_multiplier ** (-2) - (2 * clipped_count_stddev) ** (-2)
        ) ** -0.5

        adding_noise = noise_multiplier_value / noise_multiplier
        if adding_noise >= 2:
            log(
                WARNING,
                "A significant amount of noise (%s) has to be "
                "added. Consider increasing `clipped_count_stddev` or "
                "`num_sampled_clients`.",
                adding_noise,
            )

    else:
        if clipped_count_stddev is None:
            clipped_count_stddev = 0.0
        noise_multiplier_value = 0.0

    return clipped_count_stddev, noise_multiplier_value

# Local DP helper (client-side). Uses the classic Gaussian mechanism formula
# Use this when you want to reason in (ε, δ) terms per client
def add_localdp_gaussian_noise_to_params(
    model_params: Parameters, sensitivity: float, epsilon: float, delta: float
) -> Parameters:

    """Add local DP gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon,
    )
    return ndarrays_to_parameters(model_params_ndarrays)

# Local DP helper too, but gives the standard deviation directly (noise is σ).
# Use this when you want to set σ numerically (no ε/δ conversion in this function).
def add_localdp_fixed_gaussian_noise_to_params(
    model_params: Parameters, noise: float
) -> Parameters:
    """Add local DP gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        noise,
    )
    return ndarrays_to_parameters(model_params_ndarrays)
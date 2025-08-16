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
"""Local DP modifier."""



from logging import INFO
from privacy_tools.rdp_accountant import compute_rdp, get_privacy_spent
import numpy as np

from flwr.client.typing import ClientAppCallable
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.server.superlink.state import State, StateFactory
# from flwr.common.differential_privacy import (
#     add_localdp_fixed_gaussian_noise_to_params,
#     compute_clip_model_update,
# )
from fixed.differential_privacy import (
    add_localdp_fixed_gaussian_noise_to_params,
    add_localdp_gaussian_noise_to_params,
    compute_clip_model_update,
)
from flwr.common.logger import log
from flwr.common.message import Message

import json
import os

from noise_calculation.get_noise import return_epsilon

class LocalDpDynamicMod:
    # def __init__(self, clipping_norm: float, base_noise: float, max_rounds: int = 10):
    def __init__(self, clipping_norm: float, base_noise: float, 
                 max_rounds: float, dataset_size: float, target_epsilon: float = 10.0, 
                 clients_per_round: int = 2, total_clients: int = 4):
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")
        
        self.clipping_norm = clipping_norm
        self.base_noise = base_noise
        self.max_rounds = max_rounds
        self.dataset_size = dataset_size
        self.round_counter = 0
        self.client_loss_history = {}  # Track per-client loss history

        # Privacy accounting-----------------------
        self.target_epsilon = target_epsilon
        self.q = clients_per_round / total_clients  # Sampling ratio
        self.sensitivity = clipping_norm  # L2 sensitivity = clipping norm

        # implement a sampling strategy for alpha values used in RDP calc.
        # First [] creates a desn grid of orders from 1.1 to 10.9 with 0.1 increments = 99 values
        # Why? - Orders close to 1 often privde the tightest privacy bounds when converting from RDP to (epsilon, delta)-DP. 
        # Second [] adds integer orders from 12 to 63, capturing privacy behaviour at higher orders. 
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        # This array creates a 1:1 correspondence between each Renyi order and its accumulated privacy loss
        # Throughout the federated learning process, each training round contributes additional privacy loss that gets added to this cumulative tracking array.
        self.cumulative_rdp = np.zeros(len(self.orders))        

        #ensure array is writeable - log 147017639
        self.cumulative_rdp.flags.writeable = True
        # End - Privacy accounting-----------------------

        self.json_log_file = "./logs/14aug/metrics/privacy_metrics_r5.json"
        self.metrics_data = []

    def compute_dynamic_noise(self, partition_id: int, current_loss: float = None) -> float:
        """Simple dynamic noise computation based on round and optionally loss."""
        
        # Round-based decay (start high, decay over time) + the floor (30% minimum noise: + 0.3)
        # Upper bound = 1.3
        # Lower bound =~ 0.4
        round_factor = np.exp(-self.round_counter / (self.max_rounds * 0.5)) + 0.3
        
        # Loss-based adjustment (if loss is provided)
        loss_factor = 1.0
        if current_loss is not None:
            if partition_id in self.client_loss_history:
                prev_losses = self.client_loss_history[partition_id]
                if len(prev_losses) >= 2:
                    # If loss is improving rapidly, reduce noise slightly
                    improvement = (prev_losses[-1] - current_loss) / prev_losses[-1]
                    #print improvement
                    loss_factor = max(0.5, 1.0 - improvement * 0.3)  # Cap reduction

                prev_losses.append(current_loss)
                if len(prev_losses) > 3:  # Keep only last 3 losses
                    prev_losses.pop(0)
            else:
                self.client_loss_history[partition_id] = [current_loss]

        # Combine factors
        dynamic_noise = self.base_noise * round_factor * loss_factor

        # epsilon, delta = self.update_privacy_accounting(dynamic_noise)

        # Add base_noise, dynamic_noise, loss_factor to the last entry
        # current_entry = {
        #     "round": self.round_counter,
        #     "epsilon": float(epsilon),
        #     "base_noise": float(self.base_noise),
        #     "dynamic_noise": float(dynamic_noise),
        #     "loss_factor": float(loss_factor),
        #     "partition_id": float(partition_id),
        # }
        
        # # Simply append one line to file
        # with open(self.json_log_file, 'a') as f:
        #     f.write(json.dumps(current_entry) + '\n')

        print(f"[Dynamic Noise] Client {partition_id}, Round {self.round_counter}:")

        return dynamic_noise

    def __call__(self, msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
        partition_id = int(ctxt.node_config["partition-id"])
        
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        # Get server round from config if available
        fit_ins = compat.recordset_to_fitins(msg.content, keep_input=True)
        if hasattr(fit_ins, 'config') and 'current_round' in fit_ins.config:
            self.round_counter = int(fit_ins.config['current_round'])

        server_to_client_params = parameters_to_ndarrays(fit_ins.parameters)

        # Call inner app (training happens here)
        out_msg = call_next(msg, ctxt)

        if out_msg.has_error():
            return out_msg

        fit_res = compat.recordset_to_fitres(out_msg.content, keep_input=True)
        
        # Extract current loss from metrics if available
        current_loss = None
        if hasattr(fit_res, 'metrics') and 'eval_loss' in fit_res.metrics:
            current_loss = float(fit_res.metrics['eval_loss'])

        client_to_server_params = parameters_to_ndarrays(fit_res.parameters)

        # Clip the client update
        compute_clip_model_update(
            client_to_server_params,
            server_to_client_params,
            self.clipping_norm,
        )

        fit_res.parameters = ndarrays_to_parameters(client_to_server_params)

        # Compute dynamic noise
        dynamic_noise = self.compute_dynamic_noise(partition_id, current_loss)

        dynamic_noise = dynamic_noise/128

        # Add dynamic noise to model params
        fit_res.parameters = add_localdp_fixed_gaussian_noise_to_params(
            fit_res.parameters, dynamic_noise
        )

        print(f"Dataset size: {self.dataset_size}")
        noise_std = dynamic_noise / self.clipping_norm
        epsilon = return_epsilon(
            sigma=1.45,
            mode="rdp",
            dataset_size=self.dataset_size
        )

        log(INFO, f"LocalDpDynamicMod: dynamic noise {dynamic_noise:.15f} (epsilon={epsilon:.4f}) added to client {partition_id}")

        out_msg.content = compat.fitres_to_recordset(fit_res, keep_input=True)
        return out_msg


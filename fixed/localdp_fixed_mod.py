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


class LocalDpDynamicMod:
    # def __init__(self, clipping_norm: float, base_noise: float, max_rounds: int = 10):
    def __init__(self, clipping_norm: float, base_noise: float, 
                 max_rounds: float, target_epsilon: float = 10.0, 
                 clients_per_round: int = 3, total_clients: int = 4):
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        # Add persistent round tracking for logging
        self.persistent_round = 0
        self.client_round_tracker = {}  # Track rounds per client
        
        self.clipping_norm = clipping_norm
        self.base_noise = base_noise
        self.max_rounds = max_rounds
        # self.dataset_size = dataset_size
        self.round_counter = 0
        self.client_loss_history = {}  # Track per-client loss history

        # Privacy accounting-----------------------
        self.target_epsilon = target_epsilon
        self.q = clients_per_round / total_clients  # Sampling ratio
        self.sensitivity = clipping_norm  # L2 sensitivity = clipping norm

        # implement a sampling strategy for alpha values used in RDP calc.
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        # This array creates a 1:1 correspondence between each Renyi order and its accumulated privacy loss
        self.cumulative_rdp = np.zeros(len(self.orders))        

        #ensure array is writeable - log 147017639
        self.cumulative_rdp.flags.writeable = True
        # End - Privacy accounting-----------------------

        self.json_log_file = "./logs/14aug/metrics/privacy_metrics_new11.json"
        os.makedirs(os.path.dirname(self.json_log_file), exist_ok=True)
        self.metrics_data = []

    def update_privacy_accounting(self, noise_std: float):
        """Update cumulative RDP after each round."""

        # Compute noise multiplier for this round
        # = noise/clippingNorm
        noise_multiplier = noise_std / self.sensitivity

        print("Noise multiplier is:", noise_multiplier)

        # Compute RDP for this round
        round_rdp = compute_rdp(
            q=self.q,
            noise_multiplier=noise_multiplier, 
            steps=1,  # One round
            orders=self.orders
        )
        
        # Handle read-only array by creating a new writeable copy if needed
        if not self.cumulative_rdp.flags.writeable:
            self.cumulative_rdp = np.copy(self.cumulative_rdp)

        # Add to cumulative RDP
        self.cumulative_rdp += round_rdp
        
        # Convert to (ε, δ) for monitoring
        epsilon, delta, opt_order = get_privacy_spent(
            orders=self.orders,
            rdp=self.cumulative_rdp,
            target_delta=1e-2
            # target_delta=self.dataset_size
        )
        
        print(f"[Privacy] After round {self.round_counter}:\n"
              f"Epsilon (ε)={epsilon:.3f},\n"
              f"Delta (δ)={delta:.2e},\n")

        return epsilon, delta    

    def log_metrics_to_json(self, partition_id: int, dynamic_noise: float, 
                        epsilon: float, delta: float, current_loss: float = None,
                        round_factor: float = None, loss_factor: float = None):
        """Append training metrics to JSON file for plotting."""
        
        # Create metrics entry
        metrics_entry = {
            "round": self.persistent_round,
            "client_id": partition_id,
            "noise_added": float(dynamic_noise),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "current_loss": float(current_loss) if current_loss is not None else None,
            "base_noise": float(self.base_noise),
            "clipping_norm": float(self.clipping_norm),
            # "round_factor": float(round_factor) if round_factor is not None else None,
            "loss_factor": float(loss_factor) if loss_factor is not None else None,
            "noise_multiplier": float(dynamic_noise / self.clipping_norm),
        }
        
        # Append to JSON file (one line per entry)
        try:
            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(metrics_entry) + '\n')
            print(f"[JSON LOG] Metrics saved for Client {partition_id}, Round {self.persistent_round}")
        except Exception as e:
            print(f"[JSON LOG ERROR] Failed to write metrics: {e}")

    def compute_dynamic_noise(self, partition_id: int, current_loss: float = None) -> float:
        """Simple dynamic noise computation based on round and optionally loss."""
        
        # Update persistent round counter for this client
        if partition_id not in self.client_round_tracker:
            self.client_round_tracker[partition_id] = 0
        self.client_round_tracker[partition_id] = max(self.client_round_tracker[partition_id], self.round_counter)
        
        # Update global persistent round to maximum seen across all clients
        self.persistent_round = max(self.client_round_tracker.values()) if self.client_round_tracker else self.round_counter

        # Round-based decay (start high, decay over time) + the floor (30% minimum noise: + 0.3)
        # Upper bound = 1.3
        # Lower bound =~ 0.4
        round_factor = np.exp(-self.round_counter / (self.max_rounds * 0.5)) + 0.5
        
        # Loss-based adjustment (if loss is provided)
        loss_factor = 1.0
        if current_loss is not None:
            if partition_id in self.client_loss_history:
                prev_losses = self.client_loss_history[partition_id]
                if len(prev_losses) >= 2:
                    # If loss is improving rapidly, reduce noise slightly
                    improvement = (prev_losses[-1] - current_loss) / prev_losses[-1]
                    loss_factor = max(0.8, 1.2 - improvement * 0.3)  # Cap reduction
                
                prev_losses.append(current_loss)
                if len(prev_losses) > 3:  # Keep only last 3 losses
                    prev_losses.pop(0)
            else:
                self.client_loss_history[partition_id] = [current_loss]

        dynamic_noise = self.base_noise * round_factor * loss_factor

        print(f"[Dynamic Noise] Client {partition_id}, Round {self.round_counter}:\n"
              f"  base={self.base_noise:.20f}\n"
              f"  round_factor={round_factor:.20f}\n"
              f"  loss_factor={loss_factor:.20f}\n"
              f"  -> noise={dynamic_noise:.20f}")

        epsilon, delta = self.update_privacy_accounting(dynamic_noise)
        
        # Log metrics to JSON file
        self.log_metrics_to_json(
            partition_id=partition_id,
            dynamic_noise=dynamic_noise/128,  # The actual noise added after scaling
            epsilon=epsilon,
            delta=delta,
            current_loss=current_loss,
            round_factor=round_factor,
            loss_factor=loss_factor
        )

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

        # Add dynamic noise to model params
        fit_res.parameters = add_localdp_fixed_gaussian_noise_to_params(
            fit_res.parameters, dynamic_noise/128
        )
        
        log(INFO, f"LocalDpDynamicMod: dynamic noise {dynamic_noise:.4f} added to client {partition_id}")

        out_msg.content = compat.fitres_to_recordset(fit_res, keep_input=True)
        return out_msg


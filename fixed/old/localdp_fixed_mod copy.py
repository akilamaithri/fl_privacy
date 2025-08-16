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

import numpy as np

from flwr.client.typing import ClientAppCallable
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.server.superlink.state import State, StateFactory
# from flwr.common.differential_privacy import ( -- not using flwr cls
from fixed.differential_privacy import (
    add_localdp_gaussian_noise_to_params,
    add_localdp_fixed_gaussian_noise_to_params,  # -- added for fixed noise by authors
    compute_clip_model_update,
)
from flwr.common.logger import log
from flwr.common.message import Message


class LocalDpFixedMod:
    """Modifier for local differential privacy.

    This mod clips the client model updates and
    adds noise to the params before sending them to the server.

    It operates on messages of type `MessageType.TRAIN`.

    Parameters
    ----------
    clipping_norm : float
        The value of the clipping norm.
    sensitivity : float
        The sensitivity of the client model.
    epsilon : float
        The privacy budget.
        Smaller value of epsilon indicates a higher level of privacy protection.
    delta : float
        The failure probability.
        The probability that the privacy mechanism
        fails to provide the desired level of privacy.
        A smaller value of delta indicates a stricter privacy guarantee.

    epsilon_list : list[float] | None
        Per-client epsilon values when using epsilon/delta based accounting.
    delta_list : list[float] | None
        Per-client delta values when using epsilon/delta based accounting.
    noise_list : list[float] | None
        Pre-computed noise standard deviations for each client. When provided,
        ``epsilon_list`` and ``delta_list`` are ignored and ``noise_list`` is
        used to add Gaussian noise.

    Examples
    --------
    Create an instance of the local DP mod and add it to the client-side mods:

    >>> local_dp_mod = LocalDpMod( ... )
    >>> app = fl.client.ClientApp(
    >>>     client_fn=client_fn, mods=[local_dp_mod]
    >>> )
    """

    def __init__(
        self,
        clipping_norm: float,
        # epsilon_list: list[float],
        # delta_list: list[float]
        # now accepts either epsi./delta or a noise list
        epsilon_list: list[float] | None = None,
        delta_list: list[float] | None = None,
        noise_list: list[float] | None = None,
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")
        # new check
        if noise_list is None and (epsilon_list is None or delta_list is None):
            raise ValueError(
                "Either `noise_list` or both `epsilon_list` and `delta_list` must be provided."
            )
        self.clipping_norm = clipping_norm
        self.epsilon_list = epsilon_list
        self.delta_list = delta_list
        # new
        self.noise_list = noise_list 

    # noise is added
    def __call__(
        self, msg: Message, ctxt: Context, call_next: ClientAppCallable
    ) -> Message:
        """Perform local DP on the client model parameters.

        Parameters
        ----------
        msg : Message
            The message received from the server.
        ctxt : Context
            The context of the client.
        call_next : ClientAppCallable
            The callable to call the next middleware in the chain.

        Returns
        -------
        Message
            The modified message to be sent back to the server.
        """
        partition_id = ctxt.node_config["partition-id"]
        print("Partition Id is : {}".format(partition_id))
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        fit_ins = compat.recordset_to_fitins(msg.content, keep_input=True)

        # --- NEW: dynamic noise override from client config ---
        client_noise_override = None
        try:
            client_noise_override = float(fit_ins.config.get("noise_scale", None))
            if client_noise_override is not None:
                print(f"[DPMod] Overriding noise with dynamic noise_scale: {client_noise_override}")
        except Exception as e:
            print(f"[DPMod Warning] Could not read noise_scale from config: {e}")


        server_to_client_params = parameters_to_ndarrays(fit_ins.parameters)

        # Call inner app
        out_msg = call_next(msg, ctxt)

        # Check if the msg has error
        if out_msg.has_error():
            return out_msg

        fit_res = compat.recordset_to_fitres(out_msg.content, keep_input=True)

        client_to_server_params = parameters_to_ndarrays(fit_res.parameters)

        # Clip the client update
        compute_clip_model_update(
            client_to_server_params,
            server_to_client_params,
            self.clipping_norm,
        )
        log(
            INFO,
            "LocalDpMod: parameters are clipped by value: %.4f.",
            self.clipping_norm,
        )

        fit_res.parameters = ndarrays_to_parameters(client_to_server_params)

        # Add noise to model params - codex suggestion
        # fit_res.parameters = add_localdp_gaussian_noise_to_params(
        #     fit_res.parameters,
        #     epsilon=self.epsilon_list[partition_id],
        #     delta=self.delta_list[partition_id],
        #     sensitivity=self.clipping_norm, #added 2nd time
        # )
        # log(
        #     INFO,
        #     "LocalDpMod: local DP noise added with ε=%.4f, δ=%.1e",
        #     self.epsilon_list[partition_id],
        #     self.delta_list[partition_id],
        # )

        if self.noise_list is not None:
            # noise = self.noise_list[partition_id]
            # replaced above with this 👇 which allows dynamic noise
            print("Running if block for fixed noise accounting")
            noise = client_noise_override if client_noise_override is not None else self.noise_list[partition_id]
            fit_res.parameters = add_localdp_fixed_gaussian_noise_to_params(
                fit_res.parameters,
                noise,
            )
            log(
                INFO,
                "LocalDpMod: local DP noise added with \u03c3=%.4f", #sigma
                noise,
            )
        else:
            print("Running else block for epsilon/delta accounting") 
            fit_res.parameters = add_localdp_gaussian_noise_to_params(
                fit_res.parameters,
                epsilon=self.epsilon_list[partition_id],
                delta=self.delta_list[partition_id],
                sensitivity=self.clipping_norm,
            )
            log(
                INFO,
                "LocalDpMod: local DP noise added with ε=%.4f, δ=%.1e",
                self.epsilon_list[partition_id],
                self.delta_list[partition_id],
            )

        out_msg.content = compat.fitres_to_recordset(fit_res, keep_input=True)
        print(f"[Client {self.cid}] Injecting noise with sigma={sigma}")    # new addition
        return out_msg

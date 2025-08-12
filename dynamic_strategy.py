# dynamic_strategy.py
from logging import INFO, WARNING
from typing import Dict, Optional, Tuple, List

import numpy as np
import flwr as fl
from flwr.common import FitRes, Scalar, Parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg


def _bounded_scale_from_loss(
    avg_loss: float,
    *,
    alpha: float = 0.01,
    low_scale: float = 0.90,
    high_scale: float = 1.15,
) -> float:
    """Monotone bounded map: higher loss -> smaller scale (gentle).
    Returns a single scalar in [low_scale, high_scale].
    """
    # simple stable mapping; you can swap to tanh if preferred
    scale = 1.0 / (1.0 + alpha * max(0.0, float(avg_loss)))
    return float(np.clip(scale, low_scale, high_scale))


class DynamicNoiseStrategy(FedAvg):
    """FedAvg with public, bounded dynamic scaling of DP-SGD noise.

    - Uses previous-round mean client 'eval_loss' as a public signal.
    - Broadcasts one scalar `sigma_scale` via FitIns.config to all clients.
    - Keeps DP proof intact (Gaussian mech per-example; only variance changes per round).
    """

    def __init__(
        self,
        *,
        # Optional knobs; can be passed from DpArguments in federated.py
        loss_scale_lo: float = 0.90,
        loss_scale_hi: float = 1.15,
        loss_scale_alpha: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.low_scale = float(loss_scale_lo)
        self.high_scale = float(loss_scale_hi)
        self.alpha = float(loss_scale_alpha)

        self.client_losses: Dict[str, float] = {}
        self.prev_avg_loss: Optional[float] = None

        log(INFO, "DynamicNoiseStrategy initialized "
                  f"(lo={self.low_scale}, hi={self.high_scale}, alpha={self.alpha})")

    # ---- FedAvg hooks ----

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        """Return list of (ClientProxy, FitIns) with sigma_scale injected."""
        fit_confs = super().configure_fit(server_round, parameters, client_manager)

        # Compute public signal from the *previous* round
        if self.client_losses:
            avg_loss = float(np.mean(list(self.client_losses.values())))
            self.prev_avg_loss = avg_loss
            log(INFO, f"[Round {server_round}] prev mean eval_loss = {avg_loss:.4f}")
        else:
            # Bootstrap / no metrics yet
            avg_loss = self.prev_avg_loss if self.prev_avg_loss is not None else 0.0

        # Warm-up on round 1 to avoid weirdness before we have any signal
        if server_round <= 1:
            sigma_scale = 1.0
        else:
            sigma_scale = _bounded_scale_from_loss(
                avg_loss,
                alpha=self.alpha,
                low_scale=self.low_scale,
                high_scale=self.high_scale,
            )

        log(INFO, f"[Round {server_round}] sigma_scale = {sigma_scale:.4f}")

        # Inject into every client's FitIns.config
        # fit_confs is a list of (ClientProxy, FitIns)
        out = []
        for client, fit_ins in fit_confs:
            cfg = dict(fit_ins.config or {})
            cfg["sigma_scale"] = float(sigma_scale)
            fit_ins.config = cfg
            out.append((client, fit_ins))

        # Clear for next round's accumulation
        self.client_losses = {}
        return out

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Run FedAvg aggregation, then collect public signals from clients."""
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # Collect public signals (client eval_loss) for *next* round
        for client, fit_res in results:
            try:
                if fit_res.metrics is not None and "eval_loss" in fit_res.metrics:
                    self.client_losses[client.cid] = float(fit_res.metrics["eval_loss"])
                else:
                    log(WARNING, f"[Round {server_round}] Client {client.cid} did not report 'eval_loss'; "
                                 f"metrics keys: {list((fit_res.metrics or {}).keys())}")
            except Exception as e:
                log(WARNING, f"[Round {server_round}] Failed reading metrics from client {client.cid}: {e}")

        return agg_params, agg_metrics

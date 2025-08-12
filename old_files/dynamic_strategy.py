from logging import INFO
from flwr.common.logger import log
from typing import Dict, Optional, Tuple, List
import flwr as fl
from flwr.common import FitRes, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.criterion import Criterion
import numpy as np

class DynamicNoiseStrategy(FedAvg):
    """
    Custom strategy that dynamically adjusts noise based on client loss.
    """
    print("Using DynamicNoiseStrategy with FedAvg base strategy")
    #changed float value from 0.5 to 0.1 8aug
    def __init__(self, noise_base: float = 0.0001, alpha: float = 1.0, **kwargs):
        print("Initializing DynamicNoiseStrategy with noise_base:", noise_base, "and alpha:", alpha)
        super().__init__(**kwargs)
        self.noise_base = noise_base
        self.alpha = alpha
        self.client_losses = {} # Store loss from the previous round

    # Configure the fit for each client, adjusting noise based on previous losses
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)

        # Handle the first round separately
        if server_round == 1:
            log(INFO, f"First round of training, setting default noise scale for all clients.")
            for _, fit_ins in fit_configurations:
                # Set a default noise scale for the first round
                fit_ins.config["noise_scale"] = self.noise_base
        else:
            # log(INFO, f"Aggregating fit for server round {server_round} with results: {results}")
            # The rest of your existing logic for dynamic noise
            avg_loss = np.mean(list(self.client_losses.values())) if self.client_losses else 0
            log(INFO, f"Average loss from previous round: {avg_loss}")
            for client_proxy, fit_ins in fit_configurations:
                log(INFO, f"Configuring client {client_proxy.cid} with dynamic noise scale.")
                client_id = client_proxy.cid
                loss_for_client = self.client_losses.get(client_id, avg_loss)
                noise_scale = self.noise_base * (1 - (loss_for_client / (avg_loss + 5e-6))) # 8aug changed from 1e-6
                noise_scale = max(0.1, min(0.5, noise_scale)) # was 1.5 -min
                fit_ins.config["noise_scale"] = noise_scale
        
        self.client_losses = {} # Clear losses for the next round
        return fit_configurations

    # Aggregate the client updates and store the losses
    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
    #     failures: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """
    #     Aggregate the client updates and store the losses.
    #     """
    #     log(INFO, f"Aggregating fit for server round {server_round} with results: {results}")
    #     # Call FedAvg's aggregation logic
    #     aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

    #     # Extract client losses and store them for the next round
    #     for client, fit_res in results:
    #         client_id = client.cid
    #         if 'eval_loss' in fit_res.metrics:
    #             self.client_losses[client_id] = fit_res.metrics['eval_loss']

    #     return aggregated_parameters, metrics
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        log(INFO, f"Aggregating fit for server round {server_round}")

        # Call FedAvg's aggregation logic first
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # log(INFO, f"Results received: {results}") 

        # Extract and store client losses for the next round
        for client, fit_res in results:
            client_id = client.cid
            # Check if the loss metric exists and store it
            if 'eval_loss' in fit_res.metrics:
                self.client_losses[client_id] = fit_res.metrics['eval_loss']
                log(INFO, f"Client {client_id} reported 'eval_loss': {fit_res.metrics['eval_loss']}") #8aug
            else:
                # Fallback to training loss if 'eval_loss' is not available
                # You'll need to make sure your client's fit method returns this
                # This is a good place to debug and check what metrics are actually returned
                log(WARNING, f"Client {client_id} did not report 'eval_loss'. Metrics received: {fit_res.metrics}")

        log(INFO, f"Losses collected for round {server_round}: {self.client_losses}")
        
        return aggregated_parameters, aggregated_metrics
def compute_dynamic_noise(self, partition_id: int, current_loss: float = None) -> float:
    """Enhanced dynamic noise computation with more client-specific factors."""
    
    # Update client-specific round tracking
    if partition_id not in self.client_round_tracker:
        self.client_round_tracker[partition_id] = 0
    self.client_round_tracker[partition_id] += 1  # Increment for this specific client
    
    # Use client's individual round count for round factor
    client_rounds = self.client_round_tracker[partition_id]
    
    # Update global persistent round to maximum seen across all clients
    self.persistent_round = max(self.client_round_tracker.values()) if self.client_round_tracker else self.round_counter

    # CLIENT-SPECIFIC round-based decay using individual client rounds
    round_factor = np.exp(-client_rounds / (self.max_rounds * 0.5)) + 0.5
    
    # ENHANCED Loss-based adjustment with more sophisticated logic
    loss_factor = 1.0
    if current_loss is not None:
        if partition_id in self.client_loss_history:
            prev_losses = self.client_loss_history[partition_id]
            
            if len(prev_losses) >= 1:
                # Calculate loss trend
                recent_loss = prev_losses[-1]
                loss_ratio = current_loss / recent_loss if recent_loss > 0 else 1.0
                
                # If loss is decreasing (improving), reduce noise more aggressively
                if loss_ratio < 1.0:  # Loss improved
                    improvement_rate = (recent_loss - current_loss) / recent_loss
                    loss_factor = max(0.6, 1.0 - improvement_rate * 0.8)
                else:  # Loss increased or stayed same
                    degradation_rate = (current_loss - recent_loss) / recent_loss
                    loss_factor = min(1.5, 1.0 + degradation_rate * 0.5)
            
            # MORE CLIENT-SPECIFIC: Consider loss variance (stability)
            if len(prev_losses) >= 3:
                loss_variance = np.var(prev_losses)
                stability_factor = 1.0 / (1.0 + loss_variance * 10)  # Higher variance = more noise
                loss_factor *= (0.8 + 0.4 * stability_factor)
                
            prev_losses.append(current_loss)
            if len(prev_losses) > 5:  # Keep more history for better analysis
                prev_losses.pop(0)
        else:
            self.client_loss_history[partition_id] = [current_loss]
            # New clients get higher noise initially
            loss_factor = 1.2

    # ADDITIONAL CLIENT-SPECIFIC FACTORS
    
    # 1. Client performance rank factor (compare with other clients)
    performance_factor = 1.0
    if len(self.client_loss_history) > 1 and current_loss is not None:
        all_recent_losses = []
        for cid, losses in self.client_loss_history.items():
            if losses:
                all_recent_losses.append(losses[-1])
        
        if all_recent_losses:
            # Clients performing worse get more noise (more privacy protection)
            client_rank_percentile = (sorted(all_recent_losses).index(current_loss) + 1) / len(all_recent_losses)
            performance_factor = 0.8 + 0.4 * client_rank_percentile

    # 2. Client-specific random factor for additional differentiation
    np.random.seed(partition_id + client_rounds)  # Deterministic per client per round
    client_random_factor = 0.9 + 0.2 * np.random.random()  # Between 0.9 and 1.1

    # Combine all factors
    dynamic_noise = (self.base_noise * round_factor * loss_factor * 
                    performance_factor * client_random_factor)

    print(f"[Enhanced Dynamic Noise] Client {partition_id}, Local Round {client_rounds}:")
    print(f"  base={self.base_noise:.4f}")
    print(f"  round_factor={round_factor:.4f} (client-specific)")
    print(f"  loss_factor={loss_factor:.4f}")
    print(f"  performance_factor={performance_factor:.4f}")
    print(f"  client_random_factor={client_random_factor:.4f}")
    print(f"  -> final_noise={dynamic_noise:.4f}")

    epsilon, delta = self.update_privacy_accounting(dynamic_noise)
    
    # Log metrics to JSON file
    self.log_metrics_to_json(
        partition_id=partition_id,
        dynamic_noise=dynamic_noise/128,
        epsilon=epsilon,
        delta=delta,
        current_loss=current_loss,
        round_factor=round_factor,
        loss_factor=loss_factor
    )

    return dynamic_noise
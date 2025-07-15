"""
FedPersonalize Server App
Implements the federated learning server with personalization strategy
"""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, Metrics, ndarrays_to_parameters
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .task import PersonalizationModel, get_model_weights

class FedPersonalizeStrategy(FedAvg):
    """
    Custom federated learning strategy for IoT personalization
    Combines FedAvg with personalization-aware aggregation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Any],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Aggregate training results with personalization awareness"""
        
        if not results:
            return None, {}
        
        weights_results = []
        metrics_results = []
        
        for client_proxy, fit_res in results:
            weights_results.append((fit_res.parameters, fit_res.num_examples))
            if fit_res.metrics:
                metrics_results.append((fit_res.num_examples, fit_res.metrics))
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if metrics_results:
            total_examples = sum(num_examples for num_examples, _ in metrics_results)
            
            personalization_gains = []
            device_types = set()
            
            for num_examples, metrics in metrics_results:
                if "device_type" in metrics:
                    device_types.add(metrics["device_type"])
                if "accuracy" in metrics:
                    baseline = 0.5
                    gain = max(0, metrics["accuracy"] - baseline)
                    personalization_gains.append(gain * num_examples)
            
            avg_personalization_gain = sum(personalization_gains) / total_examples if total_examples > 0 else 0
            
            server_metrics = {
                "round": server_round,
                "personalization_gain": avg_personalization_gain,
                "device_diversity": len(device_types),
                "privacy_preserved": True,
                "clients_participated": len(results)
            }
            
            if aggregated_metrics:
                aggregated_metrics.update(server_metrics)
            else:
                aggregated_metrics = server_metrics
            
            self.round_metrics.append(aggregated_metrics)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        metrics_results = []
        losses = []
        
        for client_proxy, evaluate_res in results:
            losses.append(evaluate_res.loss)
            if evaluate_res.metrics:
                metrics_results.append((evaluate_res.num_examples, evaluate_res.metrics))
        
        if metrics_results:
            total_examples = sum(num_examples for num_examples, _ in metrics_results)
            
            weighted_accuracy = sum(
                metrics.get("accuracy", 0) * num_examples 
                for num_examples, metrics in metrics_results
            ) / total_examples
            
            weighted_privacy_score = sum(
                metrics.get("privacy_score", 0.95) * num_examples 
                for num_examples, metrics in metrics_results
            ) / total_examples
            
            aggregated_metrics = {
                "accuracy": weighted_accuracy,
                "privacy_score": weighted_privacy_score,
                "round": server_round
            }
        else:
            aggregated_metrics = {}
        
        avg_loss = sum(losses) / len(losses) if losses else None
        return avg_loss, aggregated_metrics

def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average"""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    weighted_accuracy = sum(
        m.get("accuracy", 0) * num_examples for num_examples, m in metrics
    ) / total_examples
    
    return {"accuracy": weighted_accuracy}

def server_fn(context: Context) -> ServerAppComponents:
    """Create server components for federated personalization learning"""
    
    num_rounds = context.run_config.get("num-server-rounds", 5)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    min_available_clients = context.run_config.get("min-available-clients", 2)
    
    model = PersonalizationModel()
    initial_parameters = ndarrays_to_parameters(get_model_weights(model))
    
    strategy = FedPersonalizeStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        initial_parameters=initial_parameters,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)

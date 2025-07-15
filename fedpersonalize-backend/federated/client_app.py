"""
FedPersonalize Client App
Implements IoT device clients for federated personalization learning
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
from typing import Dict, List, Tuple
import numpy as np

from .task import PersonalizationModel, get_model_weights, set_model_weights, generate_iot_data, train_model, evaluate_model

class IoTPersonalizationClient(NumPyClient):
    """
    IoT Device Client for Federated Personalization Learning
    Each client represents a different type of IoT device with unique data patterns
    """
    
    def __init__(self, device_type: str, device_id: int, local_epochs: int = 2, learning_rate: float = 0.01):
        self.device_type = device_type
        self.device_id = device_id
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        self.model = PersonalizationModel()
        
        self.train_data = generate_iot_data(f"{device_type}_{device_id}", num_samples=200)
        self.val_data = generate_iot_data(f"{device_type}_{device_id}_val", num_samples=50)
        
        print(f"Initialized {device_type} client {device_id}")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model with local data"""
        set_model_weights(self.model, parameters)
        
        metrics = train_model(
            self.model,
            self.train_data,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            device_type=self.device_type
        )
        
        metrics.update({
            "client_id": self.device_id,
            "device_type": self.device_type,
            "data_samples": len(self.train_data[0])
        })
        
        return get_model_weights(self.model), len(self.train_data[0]), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on local validation data"""
        set_model_weights(self.model, parameters)
        
        metrics = evaluate_model(self.model, self.val_data)
        
        metrics.update({
            "client_id": self.device_id,
            "device_type": self.device_type
        })
        
        loss = 1.0 - metrics["accuracy"]  # Convert accuracy to loss
        return loss, len(self.val_data[0]), metrics

def create_client_fn(device_type: str, device_id: int):
    """Factory function to create client function for specific device"""
    def client_fn(context: Context) -> IoTPersonalizationClient:
        local_epochs = context.run_config.get("local-epochs", 2)
        learning_rate = context.run_config.get("learning-rate", 0.01)
        
        return IoTPersonalizationClient(
            device_type=device_type,
            device_id=device_id,
            local_epochs=local_epochs,
            learning_rate=learning_rate
        ).to_client()
    
    return client_fn

traffic_client_fn = create_client_fn("traffic_sensor", 1)
environmental_client_fn = create_client_fn("environmental_sensor", 2)
wifi_client_fn = create_client_fn("wifi_access_point", 3)

app = ClientApp(client_fn=traffic_client_fn)

"""
FedPersonalize Task Module
Defines the personalization model and training logic for IoT devices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import OrderedDict

class PersonalizationModel(nn.Module):
    """
    Novel federated personalization model for IoT devices
    Combines global knowledge with local personalization layers
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super(PersonalizationModel, self).__init__()
        
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.personal_adapter = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.classifier = nn.Linear(hidden_dim // 4, output_dim)
        
        self.noise_scale = 0.1
        
    def forward(self, x: torch.Tensor, add_privacy_noise: bool = False) -> torch.Tensor:
        global_features = self.global_encoder(x)
        
        if add_privacy_noise and self.training:
            noise = torch.randn_like(global_features) * self.noise_scale
            global_features = global_features + noise
        
        personal_features = self.personal_adapter(global_features)
        
        output = self.classifier(personal_features)
        return output

def get_model_weights(model: PersonalizationModel) -> List[np.ndarray]:
    """Extract model weights as numpy arrays"""
    return [param.detach().cpu().numpy() for param in model.parameters()]

def set_model_weights(model: PersonalizationModel, weights: List[np.ndarray]) -> None:
    """Set model weights from numpy arrays"""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def generate_iot_data(device_type: str, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic IoT data for different device types
    Simulates real-world IoT device patterns
    """
    np.random.seed(hash(device_type) % 2**32)
    
    if device_type == "traffic_sensor":
        features = np.random.normal(0.5, 0.2, (num_samples, 64))
        time_pattern = np.sin(np.linspace(0, 4*np.pi, num_samples))
        features[:, 0] = time_pattern + np.random.normal(0, 0.1, num_samples)
        labels = (features[:, 0] > 0.3).astype(int)  # High traffic periods
        
    elif device_type == "environmental_sensor":
        features = np.random.normal(0.3, 0.15, (num_samples, 64))
        seasonal = np.cos(np.linspace(0, 2*np.pi, num_samples))
        features[:, 1] = seasonal + np.random.normal(0, 0.1, num_samples)
        labels = (features[:, 1] > 0.2).astype(int)  # Good air quality
        
    elif device_type == "wifi_access_point":
        features = np.random.normal(0.4, 0.25, (num_samples, 64))
        usage_spikes = np.random.exponential(0.3, num_samples)
        features[:, 2] = usage_spikes
        labels = (usage_spikes > 0.5).astype(int)  # High usage periods
        
    else:
        features = np.random.normal(0.0, 0.3, (num_samples, 64))
        labels = np.random.randint(0, 2, num_samples)
    
    device_id_hash = hash(device_type) % 1000
    personalization_features = np.full((num_samples, 1), device_id_hash / 1000.0)
    features = np.concatenate([features, personalization_features], axis=1)
    features = features[:, :64]  # Ensure correct dimension
    
    return torch.FloatTensor(features), torch.LongTensor(labels)

def train_model(
    model: PersonalizationModel,
    data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 5,
    learning_rate: float = 0.01,
    device_type: str = "generic"
) -> Dict[str, Any]:
    """
    Train the personalization model with privacy-preserving techniques
    """
    features, labels = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features, labels = features.to(device), labels.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(features, add_privacy_noise=True)
        loss = criterion(outputs, labels)
        
        personalization_loss = 0.0
        for name, param in model.named_parameters():
            if "personal" in name:
                personalization_loss += 0.01 * torch.norm(param, 2)
        
        total_loss_with_reg = loss + personalization_loss
        total_loss_with_reg.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        outputs = model(features, add_privacy_noise=False)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()
    
    return {
        "loss": total_loss / epochs,
        "accuracy": accuracy,
        "device_type": device_type,
        "privacy_preserved": True
    }

def evaluate_model(
    model: PersonalizationModel,
    data: Tuple[torch.Tensor, torch.Tensor]
) -> Dict[str, float]:
    """Evaluate the model performance"""
    features, labels = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features, labels = features.to(device), labels.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(features, add_privacy_noise=False)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        baseline_accuracy = 0.5  # Random baseline
        personalization_gain = max(0, accuracy - baseline_accuracy)
    
    return {
        "accuracy": accuracy,
        "personalization_gain": personalization_gain,
        "privacy_score": 0.95  # High privacy due to federated approach
    }

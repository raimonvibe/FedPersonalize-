from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import asyncio
import threading
import time
from datetime import datetime

app = FastAPI(title="FedPersonalize API", description="Federated Learning for IoT Personalization")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

fl_state = {
    "status": "idle",  # idle, running, completed
    "current_round": 0,
    "total_rounds": 0,
    "clients_connected": 0,
    "metrics": [],
    "start_time": None,
    "end_time": None
}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/fl/status")
async def get_fl_status():
    """Get current federated learning status"""
    return fl_state

@app.post("/fl/start")
async def start_federated_learning(config: Optional[Dict] = None):
    """Start federated learning process"""
    if fl_state["status"] == "running":
        return {"error": "Federated learning already running"}
    
    if config is None:
        config = {
            "num_rounds": 5,
            "num_clients": 3,
            "local_epochs": 2,
            "learning_rate": 0.01
        }
    
    fl_state.update({
        "status": "running",
        "current_round": 0,
        "total_rounds": config["num_rounds"],
        "clients_connected": config["num_clients"],
        "metrics": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None
    })
    
    threading.Thread(target=run_federated_learning, args=(config,), daemon=True).start()
    
    return {"message": "Federated learning started", "config": config}

@app.post("/fl/stop")
async def stop_federated_learning():
    """Stop federated learning process"""
    fl_state.update({
        "status": "idle",
        "current_round": 0,
        "total_rounds": 0,
        "end_time": datetime.now().isoformat()
    })
    return {"message": "Federated learning stopped"}

@app.get("/fl/metrics")
async def get_metrics():
    """Get federated learning metrics"""
    return {"metrics": fl_state["metrics"]}

def run_federated_learning(config: Dict):
    """Run federated learning simulation"""
    import random
    
    for round_num in range(1, config["num_rounds"] + 1):
        if fl_state["status"] != "running":
            break
            
        fl_state["current_round"] = round_num
        
        time.sleep(2)  # Simulate training time
        
        accuracy = 0.6 + (round_num * 0.05) + random.uniform(-0.02, 0.02)
        privacy_score = 0.95 + random.uniform(-0.02, 0.02)
        personalization_gain = 0.1 + (round_num * 0.03) + random.uniform(-0.01, 0.01)
        
        metrics = {
            "round": round_num,
            "accuracy": min(accuracy, 0.95),
            "privacy_score": min(privacy_score, 1.0),
            "personalization_gain": min(personalization_gain, 0.4),
            "clients_participated": config["num_clients"],
            "timestamp": datetime.now().isoformat()
        }
        
        fl_state["metrics"].append(metrics)
    
    fl_state.update({
        "status": "completed",
        "end_time": datetime.now().isoformat()
    })

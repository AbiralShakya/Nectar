import os, time, threading, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List, Optional
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import pynvml
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class SysState:
    temperature: float       # GPU temperature in °C
    power: float             # GPU power draw in Watts
    utilization: float       # GPU utilization (0–100%)
    thermal_state: str       # "cool", "warm", "hot", or "critical"
    expert_priorities: Dict[str, float]  # penalty weights per expert

@dataclass
class ThermalState:
    """Thermal state information for a GPU."""
    temperature: float
    power_watt: float
    memory_utilization: float
    compute_utilization: float
    timestamp: float
    thermal_throttle_level: float = 0.0  # 0.0 = no throttling, 1.0 = max throttling

class ThermalSignalGenerator:
    def __init__(self, device_id=0, update_interval=0.5):
        self.device_id = device_id
        self.update_interval = update_interval
        self.thermal_state = "cool"
        self.expert_priorities = {}
        self.lock = threading.Lock()

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self._start_background_update()

    def _start_background_update(self):
        thread = threading.Thread(target=self._update_loop, daemon=True)
        thread.start()

    def _update_loop(self):
        while True:
            with self.lock:
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                if temp > 85 or power > 250:
                    self.thermal_state = "critical"
                elif temp > 75:
                    self.thermal_state = "hot"
                elif temp > 60:
                    self.thermal_state = "warm"
                else:
                    self.thermal_state = "cool"
                self._update_expert_priorities()
            time.sleep(self.update_interval)

    def _update_expert_priorities(self):
        decay = {"cool": 0.0, "warm": 0.1, "hot": 0.2, "critical": 0.5}.get(self.thermal_state, 0.0)
        self.expert_priorities = {str(k): -decay * k for k in range(32)}

    def get_expert_priorities(self):
        with self.lock:
            return self.expert_priorities.copy()

    @property
    def expert_profiles(self):
        return {str(i): {"energy_cost": 1.0 + 0.05 * i} for i in range(32)}

class ThermalSignalProcessor:
    """
    Advanced thermal signal processing for multi-GPU systems.
    Handles thermal imbalances and provides thermal-aware routing signals.
    """
    def __init__(self, num_gpus: int, 
                 thermal_threshold: float = 85.0,
                 power_threshold: float = 350.0,
                 history_window: int = 100):
        self.num_gpus = num_gpus
        self.thermal_threshold = thermal_threshold
        self.power_threshold = power_threshold
        self.history_window = history_window
        
        # Thermal state history for each GPU
        self.thermal_history = {i: deque(maxlen=history_window) for i in range(num_gpus)}
        
        # Thermal gradient tracking
        self.thermal_gradients = {i: deque(maxlen=50) for i in range(num_gpus)}
        
        # Thermal imbalance detection
        self.imbalance_threshold = 10.0  # 10°C difference triggers rebalancing
        self.last_rebalance_time = 0.0
        self.rebalance_cooldown = 30.0  # 30 seconds between rebalancing
        
        # Thermal prediction model
        self.thermal_predictor = ThermalPredictor(num_gpus)
        
    def update_thermal_state(self, gpu_id: int, thermal_state: ThermalState):
        """Update thermal state for a specific GPU."""
        self.thermal_history[gpu_id].append(thermal_state)
        
        # Calculate thermal gradient
        if len(self.thermal_history[gpu_id]) > 1:
            prev_temp = self.thermal_history[gpu_id][-2].temperature
            current_temp = thermal_state.temperature
            gradient = current_temp - prev_temp
            self.thermal_gradients[gpu_id].append(gradient)
    
    def get_thermal_imbalance_score(self) -> float:
        """Calculate thermal imbalance across all GPUs."""
        if not all(self.thermal_history.values()):
            return 0.0
        
        current_temps = []
        for gpu_id in range(self.num_gpus):
            if self.thermal_history[gpu_id]:
                current_temps.append(self.thermal_history[gpu_id][-1].temperature)
        
        if not current_temps:
            return 0.0
        
        mean_temp = np.mean(current_temps)
        max_temp = np.max(current_temps)
        min_temp = np.min(current_temps)
        
        # Imbalance score based on temperature spread
        temp_spread = max_temp - min_temp
        imbalance_score = temp_spread / (mean_temp + 1e-8)
        
        return imbalance_score
    
    def get_thermal_routing_signals(self) -> Dict[str, Any]:
        """Get thermal-aware routing signals for MoE system."""
        current_time = time.time()
        
        # Get current thermal states
        current_states = {}
        for gpu_id in range(self.num_gpus):
            if self.thermal_history[gpu_id]:
                current_states[gpu_id] = self.thermal_history[gpu_id][-1]
        
        if not current_states:
            return self._get_default_signals()
        
        # Calculate thermal metrics
        temps = [state.temperature for state in current_states.values()]
        powers = [state.power_watt for state in current_states.values()]
        
        mean_temp = np.mean(temps)
        max_temp = np.max(temps)
        min_temp = np.min(temps)
        temp_spread = max_temp - min_temp
        
        # Thermal imbalance detection
        imbalance_detected = temp_spread > self.imbalance_threshold
        
        # Thermal throttling levels
        throttle_levels = {}
        for gpu_id, state in current_states.items():
            if state.temperature > self.thermal_threshold:
                throttle_factor = (state.temperature - self.thermal_threshold) / 20.0  # 20°C range
                throttle_levels[gpu_id] = min(1.0, throttle_factor)
            else:
                throttle_levels[gpu_id] = 0.0
        
        # Thermal prediction
        predicted_temps = self.thermal_predictor.predict_temperatures(current_states)
        
        # Routing recommendations
        routing_signals = {
            'thermal_imbalance_detected': imbalance_detected,
            'imbalance_score': self.get_thermal_imbalance_score(),
            'mean_temperature': mean_temp,
            'max_temperature': max_temp,
            'min_temperature': min_temp,
            'temperature_spread': temp_spread,
            'throttle_levels': throttle_levels,
            'predicted_temperatures': predicted_temps,
            'thermal_capacity_scores': self._calculate_thermal_capacity_scores(current_states),
            'rebalancing_needed': self._should_rebalance(current_time, imbalance_detected),
            'timestamp': current_time
        }
        
        return routing_signals
    
    def _calculate_thermal_capacity_scores(self, current_states: Dict[int, ThermalState]) -> Dict[int, float]:
        """Calculate thermal capacity scores for each GPU (higher = more capacity)."""
        capacity_scores = {}
        
        for gpu_id, state in current_states.items():
            # Base capacity (lower temp = higher capacity)
            temp_score = max(0.0, 1.0 - (state.temperature - 30.0) / 60.0)
            
            # Power headroom
            power_score = max(0.0, 1.0 - state.power_watt / self.power_threshold)
            
            # Memory pressure impact
            memory_score = max(0.0, 1.0 - state.memory_utilization)
            
            # Thermal gradient (negative gradient = cooling = good)
            gradient_score = 0.5
            if self.thermal_gradients[gpu_id]:
                recent_gradients = list(self.thermal_gradients[gpu_id])[-10:]
                avg_gradient = np.mean(recent_gradients)
                if avg_gradient < 0:  # Cooling
                    gradient_score = 1.0
                elif avg_gradient > 2.0:  # Heating rapidly
                    gradient_score = 0.0
            
            # Combined capacity score
            capacity_scores[gpu_id] = (
                temp_score * 0.4 + 
                power_score * 0.3 + 
                memory_score * 0.2 + 
                gradient_score * 0.1
            )
        
        return capacity_scores
    
    def _should_rebalance(self, current_time: float, imbalance_detected: bool) -> bool:
        """Determine if thermal rebalancing is needed."""
        if not imbalance_detected:
            return False
        
        # Check cooldown period
        if current_time - self.last_rebalance_time < self.rebalance_cooldown:
            return False
        
        self.last_rebalance_time = current_time
        return True
    
    def _get_default_signals(self) -> Dict[str, Any]:
        """Get default signals when no thermal data is available."""
        return {
            'thermal_imbalance_detected': False,
            'imbalance_score': 0.0,
            'mean_temperature': 50.0,
            'max_temperature': 50.0,
            'min_temperature': 50.0,
            'temperature_spread': 0.0,
            'throttle_levels': {i: 0.0 for i in range(self.num_gpus)},
            'predicted_temperatures': {i: 50.0 for i in range(self.num_gpus)},
            'thermal_capacity_scores': {i: 0.8 for i in range(self.num_gpus)},
            'rebalancing_needed': False,
            'timestamp': time.time()
        }

class ThermalPredictor(nn.Module):
    """
    Neural network for predicting thermal behavior.
    Helps anticipate thermal issues before they occur.
    """
    def __init__(self, num_gpus: int, hidden_dim: int = 64):
        super().__init__()
        self.num_gpus = num_gpus
        
        # Input features: temp, power, memory_util, compute_util, time_delta
        input_dim = 5 * num_gpus
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_gpus)  # Predict temperature for each GPU
        )
        
        # Temperature history for each GPU
        self.temp_history = {i: deque(maxlen=20) for i in range(num_gpus)}
        
    def forward(self, thermal_states: Dict[int, ThermalState]) -> Dict[int, float]:
        """Predict temperatures for next time step."""
        if not thermal_states:
            return {i: 50.0 for i in range(self.num_gpus)}
        
        # Prepare input features
        features = []
        current_time = time.time()
        
        for gpu_id in range(self.num_gpus):
            if gpu_id in thermal_states:
                state = thermal_states[gpu_id]
                features.extend([
                    state.temperature,
                    state.power_watt,
                    state.memory_utilization,
                    state.compute_utilization,
                    current_time - state.timestamp
                ])
            else:
                # Default values if no data
                features.extend([50.0, 200.0, 0.5, 0.5, 0.0])
        
        # Predict
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        predictions = self.predictor(features_tensor).squeeze(0)
        
        return {i: predictions[i].item() for i in range(self.num_gpus)}
    
    def predict_temperatures(self, thermal_states: Dict[int, ThermalState]) -> Dict[int, float]:
        """Predict temperatures (wrapper for forward pass)."""
        return self.forward(thermal_states)
    
    def update_history(self, gpu_id: int, temperature: float):
        """Update temperature history for training."""
        self.temp_history[gpu_id].append(temperature)

class ThermalAwareRouter:
    """
    Thermal-aware router that adjusts routing based on thermal signals.
    """
    def __init__(self, num_experts: int, num_gpus: int):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.experts_per_gpu = num_experts // num_gpus
        
        # Thermal signal processor
        self.thermal_processor = ThermalSignalProcessor(num_gpus)
        
        # Expert placement (expert_id -> gpu_id)
        self.expert_placement = self._default_placement()
        
        # Thermal routing history
        self.routing_history = deque(maxlen=1000)
        
    def _default_placement(self) -> Dict[int, int]:
        """Default round-robin expert placement."""
        placement = {}
        for expert_id in range(self.num_experts):
            placement[expert_id] = expert_id % self.num_gpus
        return placement
    
    def update_thermal_state(self, gpu_id: int, thermal_state: ThermalState):
        """Update thermal state for routing decisions."""
        self.thermal_processor.update_thermal_state(gpu_id, thermal_state)
    
    def get_thermal_adjusted_routing(self, base_routing: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Adjust routing based on thermal signals."""
        thermal_signals = self.thermal_processor.get_thermal_routing_signals()
        
        # If no thermal imbalance, return base routing
        if not thermal_signals['thermal_imbalance_detected']:
            return base_routing
        
        # Get thermal capacity scores
        capacity_scores = thermal_signals['thermal_capacity_scores']
        
        # Adjust routing to favor cooler GPUs
        adjusted_routing = {}
        
        for token_id, expert_ids in base_routing.items():
            # Score each expert based on GPU thermal capacity
            expert_scores = []
            for expert_id in expert_ids:
                gpu_id = self.expert_placement[expert_id]
                capacity_score = capacity_scores.get(gpu_id, 0.5)
                expert_scores.append((expert_id, capacity_score))
            
            # Sort by capacity score (higher = better)
            expert_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top experts with thermal preference
            adjusted_experts = [expert_id for expert_id, _ in expert_scores[:2]]
            adjusted_routing[token_id] = adjusted_experts
        
        # Log routing adjustment
        self.routing_history.append({
            'timestamp': time.time(),
            'thermal_signals': thermal_signals,
            'routing_adjusted': True
        })
        
        return adjusted_routing
    
    def should_migrate_experts(self) -> Tuple[bool, Dict[int, int]]:
        """Determine if experts should be migrated based on thermal state."""
        thermal_signals = self.thermal_processor.get_thermal_routing_signals()
        
        if not thermal_signals['rebalancing_needed']:
            return False, {}
        
        # Calculate new placement based on thermal capacity
        capacity_scores = thermal_signals['thermal_capacity_scores']
        
        # Sort experts by current usage (would need to be tracked)
        # For now, use simple round-robin with thermal preference
        new_placement = {}
        gpu_loads = [0] * self.num_gpus
        
        for expert_id in range(self.num_experts):
            # Find GPU with highest capacity and lowest load
            best_gpu = 0
            best_score = -1
            
            for gpu_id in range(self.num_gpus):
                capacity = capacity_scores.get(gpu_id, 0.5)
                load_ratio = gpu_loads[gpu_id] / self.experts_per_gpu
                score = capacity * (1.0 - load_ratio)
                
                if score > best_score:
                    best_score = score
                    best_gpu = gpu_id
            
            new_placement[expert_id] = best_gpu
            gpu_loads[best_gpu] += 1
        
        return True, new_placement
    
    def get_thermal_stats(self) -> Dict[str, Any]:
        """Get thermal statistics for monitoring."""
        thermal_signals = self.thermal_processor.get_thermal_routing_signals()
        
        return {
            'thermal_imbalance_score': thermal_signals['imbalance_score'],
            'mean_temperature': thermal_signals['mean_temperature'],
            'temperature_spread': thermal_signals['temperature_spread'],
            'throttle_levels': thermal_signals['throttle_levels'],
            'routing_adjustments': len([r for r in self.routing_history if r['routing_adjusted']]),
            'last_rebalancing': self.thermal_processor.last_rebalance_time
        }

# Utility functions for thermal testing
def create_thermal_test_scenario(num_gpus: int = 8) -> Dict[int, ThermalState]:
    """Create a test scenario with thermal imbalance."""
    thermal_states = {}
    
    # Create thermal imbalance: one GPU at 0°C, others at 90°C
    for gpu_id in range(num_gpus):
        if gpu_id == 0:
            # Cool GPU
            thermal_states[gpu_id] = ThermalState(
                temperature=30.0,
                power_watt=150.0,
                memory_utilization=0.3,
                compute_utilization=0.4,
                timestamp=time.time()
            )
        else:
            # Hot GPUs
            thermal_states[gpu_id] = ThermalState(
                temperature=85.0 + gpu_id * 2.0,  # Slight variation
                power_watt=350.0,
                memory_utilization=0.8,
                compute_utilization=0.9,
                timestamp=time.time()
            )
    
    return thermal_states

def test_thermal_awareness():
    """Test thermal awareness functionality."""
    num_gpus = 8
    num_experts = 16
    
    # Create thermal router
    thermal_router = ThermalAwareRouter(num_experts, num_gpus)
    
    # Create test scenario
    thermal_states = create_thermal_test_scenario(num_gpus)
    
    # Update thermal states
    for gpu_id, state in thermal_states.items():
        thermal_router.update_thermal_state(gpu_id, state)
    
    # Test routing adjustment
    base_routing = {i: [i % num_experts, (i + 1) % num_experts] for i in range(100)}
    adjusted_routing = thermal_router.get_thermal_adjusted_routing(base_routing)
    
    # Test expert migration
    should_migrate, new_placement = thermal_router.should_migrate_experts()
    
    # Get stats
    stats = thermal_router.get_thermal_stats()
    
    print("Thermal Test Results:")
    print(f"Imbalance Score: {stats['thermal_imbalance_score']:.3f}")
    print(f"Mean Temperature: {stats['mean_temperature']:.1f}°C")
    print(f"Temperature Spread: {stats['temperature_spread']:.1f}°C")
    print(f"Should Migrate: {should_migrate}")
    print(f"Routing Adjustments: {stats['routing_adjustments']}")
    
    return {
        'thermal_router': thermal_router,
        'adjusted_routing': adjusted_routing,
        'new_placement': new_placement,
        'stats': stats
    }

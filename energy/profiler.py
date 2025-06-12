import threading
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pynvml

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

PYNVML_AVAILABLE = True

@dataclass
class GPUMetrics:
    """Container for GPU metrics at a specific timestamp."""
    timestamp: float
    power_draw: float  # Watts
    temperature: float  # Celsius
    memory_used: int   # Bytes
    memory_total: int  # Bytes
    gpu_utilization: float  # Percentage
    memory_utilization: float  # Percentage


@dataclass
class ExpertProfile:
    """Profile data for a specific expert."""
    expert_id: str
    flops: int
    memory_footprint: int
    avg_latency: float
    energy_cost: float  # Estimated Joules
    activation_count: int
    last_updated: float


class GPUProfiler:
    """
    Continuously monitors GPU metrics and provides energy profiling for MoE experts.
    Runs in a separate thread to avoid blocking main computation.
    """
    
    def __init__(self, device_id: int = 0, poll_interval: float = 0.1):
        self.device_id = device_id
        self.poll_interval = poll_interval
        self.is_running = False
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 samples
        self.expert_profiles = {}
        self.operation_stack = []  # Stack for nested operations
        self.lock = threading.Lock()
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.nvml_available = True
            except Exception as e:
                logging.warning(f"Failed to initialize NVML: {e}")
                self.nvml_available = False
        else:
            self.nvml_available = False
            
        self.polling_thread = None
        
    def start_profiling(self):
        """Start the GPU monitoring thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logging.info("GPU profiling started")
        
    def stop_profiling(self):
        """Stop the GPU monitoring thread."""
        self.is_running = False
        if self.polling_thread:
            self.polling_thread.join(timeout=1.0)
        logging.info("GPU profiling stopped")
        
    def _polling_loop(self):
        """Main polling loop running in separate thread."""
        while self.is_running:
            try:
                metrics = self._collect_gpu_metrics()
                if metrics:
                    with self.lock:
                        self.metrics_history.append(metrics)
                time.sleep(self.poll_interval)
            except Exception as e:
                logging.error(f"Error in GPU polling: {e}")
                time.sleep(self.poll_interval)
                
    def _collect_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Collect current GPU metrics."""
        if not self.nvml_available:
            # Return dummy metrics for testing
            return GPUMetrics(
                timestamp=time.time(),
                power_draw=150.0 + np.random.normal(0, 10),
                temperature=65.0 + np.random.normal(0, 5),
                memory_used=int(4e9 + np.random.normal(0, 1e8)),
                memory_total=int(8e9),
                gpu_utilization=80.0 + np.random.normal(0, 10),
                memory_utilization=50.0 + np.random.normal(0, 5)
            )
            
        try:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = mem_info.used
            memory_total = mem_info.total
            
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util_rates.gpu
            memory_util = util_rates.memory
            
            return GPUMetrics(
                timestamp=time.time(),
                power_draw=power_draw,
                temperature=temperature,
                memory_used=memory_used,
                memory_total=memory_total,
                gpu_utilization=gpu_util,
                memory_utilization=memory_util
            )
        except Exception as e:
            logging.error(f"Failed to collect GPU metrics: {e}")
            return None
            
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get the most recent GPU metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
            
    def get_metrics_window(self, duration: float = 1.0) -> List[GPUMetrics]:
        """Get metrics from the last `duration` seconds."""
        current_time = time.time()
        cutoff_time = current_time - duration
        
        with self.lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
    def start_operation(self, operation_name: str, expert_id: Optional[str] = None):
        """Start tracking an operation (e.g., expert forward pass)."""
        operation_data = {
            'name': operation_name,
            'expert_id': expert_id,
            'start_time': time.time(),
            'start_metrics': self.get_current_metrics()
        }
        self.operation_stack.append(operation_data)
        
    def end_operation(self) -> Optional[Dict]:
        """End tracking the current operation and return profiling data."""
        if not self.operation_stack:
            return None
            
        operation_data = self.operation_stack.pop()
        end_time = time.time()
        end_metrics = self.get_current_metrics()
        
        duration = end_time - operation_data['start_time']
        
        # Calculate energy consumption during operation
        energy_consumed = 0.0
        if operation_data['start_metrics'] and end_metrics:
            avg_power = (operation_data['start_metrics'].power_draw + end_metrics.power_draw) / 2.0
            energy_consumed = avg_power * duration  # Joules
            
        result = {
            'operation': operation_data['name'],
            'expert_id': operation_data['expert_id'],
            'duration': duration,
            'energy_consumed': energy_consumed,
            'start_metrics': operation_data['start_metrics'],
            'end_metrics': end_metrics
        }
        
        # Update expert profile if applicable
        if operation_data['expert_id']:
            self._update_expert_profile(operation_data['expert_id'], result)
            
        return result
        
    def _update_expert_profile(self, expert_id: str, operation_result: Dict):
        """Update the profile for a specific expert."""
        if expert_id not in self.expert_profiles:
            self.expert_profiles[expert_id] = ExpertProfile(
                expert_id=expert_id,
                flops=0,
                memory_footprint=0,
                avg_latency=0.0,
                energy_cost=0.0,
                activation_count=0,
                last_updated=time.time()
            )
            
        profile = self.expert_profiles[expert_id]
        profile.activation_count += 1
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        profile.avg_latency = (1 - alpha) * profile.avg_latency + alpha * operation_result['duration']
        profile.energy_cost = (1 - alpha) * profile.energy_cost + alpha * operation_result['energy_consumed']
        profile.last_updated = time.time()
        
    def estimate_expert_flops(self, expert_module: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate FLOPs for an expert module.
        This is a simplified estimation - you might want to use more sophisticated methods.
        """
        total_flops = 0
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Linear):
                # For linear layers: input_size * output_size * batch_size
                in_features = module.in_features
                out_features = module.out_features
                batch_size = input[0].shape[0] if input else 1
                total_flops += in_features * out_features * batch_size
            elif isinstance(module, nn.Conv2d):
                # Simplified conv2d FLOP estimation
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = output.numel() if hasattr(output, 'numel') else 1
                total_flops += kernel_flops * output_elements
                
        # Register hooks
        hooks = []
        for module in expert_module.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(flop_count_hook))
                
        try:
            with torch.no_grad():
                expert_module(dummy_input)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
        return total_flops
        
    def get_expert_profile(self, expert_id: str) -> Optional[ExpertProfile]:
        """Get the profile for a specific expert."""
        return self.expert_profiles.get(expert_id)
        
    def get_all_expert_profiles(self) -> Dict[str, ExpertProfile]:
        """Get profiles for all experts."""
        return self.expert_profiles.copy()
        
    def save_profiles(self, filepath: str):
        """Save expert profiles to JSON file."""
        profiles_dict = {
            expert_id: asdict(profile) 
            for expert_id, profile in self.expert_profiles.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profiles_dict, f, indent=2)
            
    def load_profiles(self, filepath: str):
        """Load expert profiles from JSON file."""
        try:
            with open(filepath, 'r') as f:
                profiles_dict = json.load(f)
                
            self.expert_profiles = {
                expert_id: ExpertProfile(**profile_data)
                for expert_id, profile_data in profiles_dict.items()
            }
        except Exception as e:
            logging.error(f"Failed to load profiles: {e}")
            
    def get_power_statistics(self, duration: float = 10.0) -> Dict[str, float]:
        """Get power consumption statistics over a time window."""
        metrics = self.get_metrics_window(duration)
        if not metrics:
            return {}
            
        power_values = [m.power_draw for m in metrics]
        
        return {
            'mean_power': np.mean(power_values),
            'max_power': np.max(power_values),
            'min_power': np.min(power_values),
            'std_power': np.std(power_values),
            'total_energy': np.sum(power_values) * self.poll_interval  # Approximate
        }
        
    def __enter__(self):
        """Context manager entry."""
        self.start_profiling()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_profiling()


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the GPUProfiler
    profiler = GPUProfiler(device_id=0, poll_interval=0.1)
    
    # Start profiling
    profiler.start_profiling()
    
    # Simulate some operations
    time.sleep(1.0)
    
    # Example expert operation
    profiler.start_operation("expert_forward", "expert_0")
    time.sleep(0.5)  # Simulate computation
    result = profiler.end_operation()
    
    print("Operation result:", result)
    
    # Get current metrics
    current = profiler.get_current_metrics()
    if current:
        print(f"Current power: {current.power_draw:.2f}W, Temp: {current.temperature:.1f}°C")
    
    # Get power statistics
    stats = profiler.get_power_statistics(duration=5.0)
    print("Power statistics:", stats)
    
    # Stop profiling
    profiler.stop_profiling()
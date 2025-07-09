import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math
import random

class KernelCostModel:
    """
    Advanced kernel cost model for LLM/GPU workloads with realistic hardware characteristics.
    Models energy, latency, and thermal impact based on actual GPU behavior patterns.
    Now includes error margins and synthetic noise for robustness testing.
    """
    def __init__(self, data_path: Optional[str] = None, gpu_type: str = "A100", 
                 noise_level: float = 0.05, error_margin: float = 0.1):
        self.data = pd.DataFrame()
        self.interpolation_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.gpu_type = gpu_type
        
        # Error handling and robustness parameters
        self.noise_level = noise_level  # 5% default noise
        self.error_margin = error_margin  # 10% error margin
        self._enable_synthetic_noise = True
        self._enable_error_margins = True
        
        # Hardware-specific constants based on real GPU characteristics
        self.gpu_specs = self._get_gpu_specs(gpu_type)
        
        if data_path:
            try:
                self.data = pd.read_json(data_path)
                self.data.set_index(["op_type", "batch_size"], inplace=True)
                print(f"KernelCostModel loaded from {data_path}. Entries: {len(self.data)}")
                self.unique_op_types = self.data.index.get_level_values(0).unique().tolist()
            except (FileNotFoundError, Exception) as e:
                print(f"Error loading data: {e}. Initializing with realistic synthetic data.")
                self._initialize_realistic_data()
        else:
            print("KernelCostModel initialized with realistic synthetic data for LLM workloads.")
            self._initialize_realistic_data()

    def _add_synthetic_noise(self, value: float, operation_type: str) -> float:
        """Add synthetic noise to simulate real-world measurement variations."""
        if not self._enable_synthetic_noise:
            return value
        
        # Different noise levels for different operation types
        noise_multipliers = {
            'attention': 0.03,  # Lower noise for attention (more predictable)
            'ffn': 0.05,        # Medium noise for FFN
            'moe': 0.08,        # Higher noise for MoE (more variable)
            'quantize': 0.04,   # Medium noise for quantization
            'default': 0.05
        }
        
        # Determine noise level based on operation type
        noise_level = noise_multipliers.get('default')
        for op_pattern, noise in noise_multipliers.items():
            if op_pattern in operation_type:
                noise_level = noise
                break
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * value)
        return value + noise

    def _apply_error_margins(self, value: float, operation_type: str) -> Tuple[float, float]:
        """Apply error margins to provide confidence intervals."""
        if not self._enable_error_margins:
            return value, value
        
        # Different error margins for different operations
        margin_multipliers = {
            'attention': 0.08,   # Lower margin for attention
            'ffn': 0.12,         # Medium margin for FFN
            'moe': 0.15,         # Higher margin for MoE
            'quantize': 0.10,    # Medium margin for quantization
            'default': 0.10
        }
        
        margin_level = margin_multipliers.get('default')
        for op_pattern, margin in margin_multipliers.items():
            if op_pattern in operation_type:
                margin_level = margin
                break
        
        margin = margin_level * value
        return value - margin, value + margin

    def _get_gpu_specs(self, gpu_type: str) -> Dict:
        """Hardware specifications for different GPU types"""
        specs = {
            "A100": {
                "peak_power_w": 400,
                "memory_bandwidth_gb_s": 1935,
                "compute_throughput_tflops": 312,  # BF16
                "base_temp_c": 30,
                "thermal_throttle_temp_c": 87,
                "memory_size_gb": 80,
                "sm_count": 108,
                "error_margin_power": 0.05,    # 5% power measurement error
                "error_margin_temp": 0.02,     # 2°C temperature measurement error
                "error_margin_memory": 0.03    # 3% memory measurement error
            },
            "H100": {
                "peak_power_w": 700,
                "memory_bandwidth_gb_s": 3350,
                "compute_throughput_tflops": 989,  # BF16
                "base_temp_c": 30,
                "thermal_throttle_temp_c": 90,
                "memory_size_gb": 80,
                "sm_count": 132,
                "error_margin_power": 0.04,    # 4% power measurement error
                "error_margin_temp": 0.015,    # 1.5°C temperature measurement error
                "error_margin_memory": 0.025   # 2.5% memory measurement error
            },
            "V100": {
                "peak_power_w": 300,
                "memory_bandwidth_gb_s": 900,
                "compute_throughput_tflops": 125,  # FP16
                "base_temp_c": 30,
                "thermal_throttle_temp_c": 83,
                "memory_size_gb": 32,
                "sm_count": 80,
                "error_margin_power": 0.06,    # 6% power measurement error
                "error_margin_temp": 0.025,    # 2.5°C temperature measurement error
                "error_margin_memory": 0.04    # 4% memory measurement error
            }
        }
        return specs.get(gpu_type, specs["A100"])

    def _initialize_realistic_data(self):
        """Initialize with realistic LLM/transformer operation costs"""
        llm_ops = {
            # Attention operations - memory bandwidth bound
            "attention_qk": {"flops_per_token": 4096, "memory_intensity": 2.0, "parallelizable": True},
            "attention_av": {"flops_per_token": 4096, "memory_intensity": 1.5, "parallelizable": True},
            "attention_proj": {"flops_per_token": 16777216, "memory_intensity": 1.0, "parallelizable": True},
            "ffn_gate": {"flops_per_token": 33554432, "memory_intensity": 0.8, "parallelizable": True},
            "ffn_up": {"flops_per_token": 33554432, "memory_intensity": 0.8, "parallelizable": True},
            "ffn_down": {"flops_per_token": 33554432, "memory_intensity": 0.8, "parallelizable": True},
            "silu_gelu": {"flops_per_token": 1024, "memory_intensity": 3.0, "parallelizable": True},
            "layer_norm": {"flops_per_token": 8192, "memory_intensity": 2.5, "parallelizable": False},
            "rmsnorm": {"flops_per_token": 4096, "memory_intensity": 2.0, "parallelizable": False},
            "token_embed": {"flops_per_token": 4096, "memory_intensity": 4.0, "parallelizable": True},
            "pos_embed": {"flops_per_token": 2048, "memory_intensity": 3.0, "parallelizable": True},
            "quantize_w8a16": {"flops_per_token": 1024, "memory_intensity": 1.5, "parallelizable": True},
            "dequantize_w8a16": {"flops_per_token": 2048, "memory_intensity": 2.0, "parallelizable": True},
            "moe_router": {"flops_per_token": 2048, "memory_intensity": 1.0, "parallelizable": False},
            "expert_selection": {"flops_per_token": 512, "memory_intensity": 2.0, "parallelizable": False},
            "token_dispatch": {"flops_per_token": 256, "memory_intensity": 3.0, "parallelizable": False},
            "token_combine": {"flops_per_token": 512, "memory_intensity": 2.5, "parallelizable": False},
        }
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        all_costs = []
        for op_name, op_props in llm_ops.items():
            for batch_size in batch_sizes:
                costs = self._calculate_realistic_costs(op_name, op_props, batch_size)
                # Ensure correct types
                costs["op_type"] = str(op_name)
                costs["batch_size"] = int(batch_size)
                all_costs.append(costs)
        self.data = pd.DataFrame(all_costs)
        self.data.set_index(["op_type", "batch_size"], inplace=True)
        self.unique_op_types = list(llm_ops.keys())
        print(f"Initialized realistic kernel cost model with {len(llm_ops)} operation types")

    def _calculate_realistic_costs(self, op_name: str, op_props: Dict, batch_size: int) -> Dict[str, float]:
        """Calculate realistic costs based on hardware characteristics and operation properties"""
        
        flops_per_token = op_props["flops_per_token"]
        memory_intensity = op_props["memory_intensity"]
        is_parallelizable = op_props["parallelizable"]
        
        total_flops = flops_per_token * batch_size
        
        # Memory bandwidth utilization (bytes per token)
        memory_bytes = flops_per_token * memory_intensity * 2  # Assuming BF16 (2 bytes)
        total_memory_bytes = memory_bytes * batch_size
        
        # Compute time vs memory time
        compute_time_ms = (total_flops / (self.gpu_specs["compute_throughput_tflops"] * 1e12)) * 1000
        memory_time_ms = (total_memory_bytes / (self.gpu_specs["memory_bandwidth_gb_s"] * 1e9)) * 1000
        
        # Actual latency is dominated by the bottleneck
        base_latency_ms = max(compute_time_ms, memory_time_ms)
        
        # Batch size scaling effects
        if is_parallelizable:
            # Good parallelization but with diminishing returns due to memory bandwidth
            efficiency = min(1.0, 32 / batch_size + 0.7)  # Efficiency drops with very large batches
            latency_ms = base_latency_ms * efficiency
        else:
            # Operations that don't parallelize well (like norms, routing)
            latency_ms = base_latency_ms * math.log2(max(1, batch_size / 16)) * 0.3 + base_latency_ms
        
        # Kernel launch overhead (becomes negligible at larger batch sizes)
        launch_overhead_ms = max(0.001, 0.05 / math.sqrt(batch_size))
        latency_ms += launch_overhead_ms
        
        # Realistic energy calculation
        # Base energy from computation
        compute_energy_j = (total_flops / (self.gpu_specs["compute_throughput_tflops"] * 1e12)) * \
                          (self.gpu_specs["peak_power_w"] * 0.6)  # 60% of peak power for compute
        
        # Memory access energy
        memory_energy_j = (total_memory_bytes / (self.gpu_specs["memory_bandwidth_gb_s"] * 1e9)) * \
                         (self.gpu_specs["peak_power_w"] * 0.3)  # 30% of peak power for memory
        
        # Static power during execution
        static_energy_j = (latency_ms / 1000) * (self.gpu_specs["peak_power_w"] * 0.1)  # 10% static power
        
        total_energy_j = compute_energy_j + memory_energy_j + static_energy_j
        
        # Thermal impact modeling
        # Heat generation is proportional to power dissipation
        peak_power_w = total_energy_j / (latency_ms / 1000)
        power_density = peak_power_w / self.gpu_specs["sm_count"]  # Power per SM
        
        # Temperature rise depends on power density and duration
        temp_impact_c = power_density * 0.1 * math.sqrt(latency_ms / 1000)
        
        # Memory-bound operations generate less heat per FLOP
        if memory_time_ms > compute_time_ms:
            temp_impact_c *= 0.7
        
        # Add operation-specific thermal characteristics
        thermal_modifiers = {
            "attention": 0.9,    # Attention is memory-bound, less heat
            "ffn": 1.2,          # FFN is compute-intensive, more heat
            "norm": 0.6,         # Normalization is lightweight
            "moe": 1.1,          # MoE routing has irregular access patterns
            "quantize": 0.8,     # Quantization is memory-focused
        }
        
        for pattern, modifier in thermal_modifiers.items():
            if pattern in op_name:
                temp_impact_c *= modifier
                break
        
        # Apply synthetic noise and error margins
        energy_j = self._add_synthetic_noise(total_energy_j, op_name)
        latency_ms = self._add_synthetic_noise(latency_ms, op_name)
        temp_impact_c = self._add_synthetic_noise(temp_impact_c, op_name)
        
        return {
            "energy_joules": max(energy_j, 1e-6),
            "latency_ms": max(latency_ms, 0.001),
            "temp_impact": max(temp_impact_c, 0.001),
            "memory_gb": total_memory_bytes / (1024**3),  # Memory footprint
            "compute_utilization": min(1.0, compute_time_ms / latency_ms),
            "memory_utilization": min(1.0, memory_time_ms / latency_ms)
        }

    def get_cost(self, op_type: str, batch_size: int, 
                 current_temp: float = None, memory_pressure: float = 0.0) -> Dict[str, float]:
        """
        Enhanced cost lookup with thermal and memory pressure adjustments
        Now includes error margins and confidence intervals
        """
        # Default current_temp to 0.0 if None
        if current_temp is None:
            current_temp = 0.0
        cache_key = (op_type, batch_size)
        if cache_key in self.interpolation_cache:
            return self.interpolation_cache[cache_key]

        if op_type not in self.unique_op_types:
            print(f"Warning: Unknown op_type '{op_type}'. Returning default costs.")
            return {"energy_joules": 0.001, "latency_ms": 0.1, "temp_impact": 0.001}

        # Get base costs (with interpolation if needed)
        base_costs = self._get_base_cost(op_type, batch_size)
        
        # Apply thermal and memory pressure adjustments
        thermal_factor = self._calculate_thermal_factor(current_temp)
        memory_factor = self._calculate_memory_factor(memory_pressure)
        
        # Adjust costs based on hardware state
        adjusted_costs = {}
        for key, value in base_costs.items():
            if key == "energy_joules":
                adjusted_costs[key] = value * thermal_factor * memory_factor
            elif key == "latency_ms":
                adjusted_costs[key] = value * thermal_factor  # Temperature affects latency more
            elif key == "temp_impact":
                adjusted_costs[key] = value * thermal_factor
            else:
                adjusted_costs[key] = value
        
        # Apply error margins and get confidence intervals
        if self._enable_error_margins:
            energy_min, energy_max = self._apply_error_margins(adjusted_costs["energy_joules"], op_type)
            latency_min, latency_max = self._apply_error_margins(adjusted_costs["latency_ms"], op_type)
            temp_min, temp_max = self._apply_error_margins(adjusted_costs["temp_impact"], op_type)
            
            adjusted_costs.update({
                "energy_joules_min": energy_min,
                "energy_joules_max": energy_max,
                "latency_ms_min": latency_min,
                "latency_ms_max": latency_max,
                "temp_impact_min": temp_min,
                "temp_impact_max": temp_max,
                "confidence_level": 0.95  # 95% confidence interval
            })
        
        # Cache the result
        self.interpolation_cache[cache_key] = adjusted_costs
        
        return adjusted_costs

    def _get_base_cost(self, op_type: str, batch_size: int) -> Dict[str, float]:
        """Get base cost with interpolation if needed"""
        try:
            # Try exact match first
            return self.data.loc[(op_type, batch_size)].to_dict()
        except KeyError:
            # Interpolate between available batch sizes
            available_sizes = self.data.loc[op_type].index.tolist()
            if not available_sizes:
                return {"energy_joules": 0.001, "latency_ms": 0.1, "temp_impact": 0.001}
            
            # Find closest batch sizes
            available_sizes.sort()
            if batch_size <= available_sizes[0]:
                return self.data.loc[(op_type, available_sizes[0])].to_dict()
            elif batch_size >= available_sizes[-1]:
                return self.data.loc[(op_type, available_sizes[-1])].to_dict()
            else:
                # Linear interpolation
                for i, size in enumerate(available_sizes):
                    if size > batch_size:
                        lower_size = available_sizes[i-1]
                        upper_size = size
                        lower_cost = self.data.loc[(op_type, lower_size)].to_dict()
                        upper_cost = self.data.loc[(op_type, upper_size)].to_dict()
                        
                        # Interpolate
                        ratio = (batch_size - lower_size) / (upper_size - lower_size)
                        interpolated = {}
                        for key in lower_cost:
                            if isinstance(lower_cost[key], (int, float)):
                                interpolated[key] = lower_cost[key] + ratio * (upper_cost[key] - lower_cost[key])
                            else:
                                interpolated[key] = lower_cost[key]
                        
                        return interpolated
        
        return {"energy_joules": 0.001, "latency_ms": 0.1, "temp_impact": 0.001}

    def _calculate_thermal_factor(self, current_temp: float) -> float:
        """Calculate thermal scaling factor based on current temperature"""
        base_temp = self.gpu_specs["base_temp_c"]
        throttle_temp = self.gpu_specs["thermal_throttle_temp_c"]
        
        if current_temp <= base_temp:
            return 1.0
        elif current_temp >= throttle_temp:
            return 2.0  # Severe throttling
        else:
            # Linear scaling between base and throttle temperature
            temp_ratio = (current_temp - base_temp) / (throttle_temp - base_temp)
            return 1.0 + temp_ratio

    def _calculate_memory_factor(self, memory_pressure: float) -> float:
        """Calculate memory pressure scaling factor"""
        if memory_pressure <= 0.5:
            return 1.0
        elif memory_pressure >= 0.9:
            return 1.5  # Severe memory pressure
        else:
            # Linear scaling between 50% and 90% memory usage
            pressure_ratio = (memory_pressure - 0.5) / 0.4
            return 1.0 + 0.5 * pressure_ratio

    def get_cost_breakdown(
        self,
        op_type: str,
        batch_size: int,
        current_temp: float = None,
        memory_pressure: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get detailed cost breakdown with confidence intervals
        """
        costs = self.get_cost(op_type, batch_size, current_temp, memory_pressure)
        
        breakdown = {
            "operation": op_type,
            "batch_size": batch_size,
            "current_temperature": current_temp,
            "memory_pressure": memory_pressure,
            "base_costs": costs,
            "thermal_factor": self._calculate_thermal_factor(current_temp or 0.0),
            "memory_factor": self._calculate_memory_factor(memory_pressure),
            "gpu_specs": self.gpu_specs,
            "noise_level": self.noise_level,
            "error_margin": self.error_margin
        }
        
        return breakdown

    def get_all_op_types(self) -> List[str]:
        """Get list of all supported operation types"""
        return self.unique_op_types

    def get_thermal_safe_batch_size(self, op_type: str, current_temp: float, 
                                   max_temp_increase: float = 5.0) -> int:
        """
        Calculate the maximum safe batch size given thermal constraints
        """
        # Start with small batch size and increase until thermal limit
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            costs = self.get_cost(op_type, batch_size, current_temp)
            temp_increase = costs["temp_impact"]
            
            if temp_increase > max_temp_increase:
                # Return the previous safe batch size
                return max(1, batch_size // 2)
        
        return 1024  # Default to maximum if no thermal issues

    def enable_noise_injection(self, enable: bool = True, noise_level: float = None):
        """Enable/disable synthetic noise injection"""
        self.enable_synthetic_noise = enable
        if noise_level is not None:
            self.noise_level = noise_level

    def enable_error_margins(self, enable: bool = True, margin_level: float = None):
        """Enable/disable error margins"""
        self.enable_error_margins = enable
        if margin_level is not None:
            self.error_margin = margin_level

    def get_confidence_intervals(self, op_type: str, batch_size: int, 
                               current_temp: float = None, memory_pressure: float = 0.0) -> Dict[str, Tuple[float, float]]:
        """
        Get confidence intervals for all cost metrics
        """
        costs = self.get_cost(op_type, batch_size, current_temp, memory_pressure)
        
        intervals = {}
        for key in ["energy_joules", "latency_ms", "temp_impact"]:
            if f"{key}_min" in costs and f"{key}_max" in costs:
                intervals[key] = (costs[f"{key}_min"], costs[f"{key}_max"])
            else:
                # If no error margins, create symmetric intervals
                value = costs[key]
                margin = value * self.error_margin
                intervals[key] = (value - margin, value + margin)
        
        return intervals
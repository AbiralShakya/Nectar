import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math

class KernelCostModel:
    """
    Advanced kernel cost model for LLM/GPU workloads with realistic hardware characteristics.
    Models energy, latency, and thermal impact based on actual GPU behavior patterns.
    """
    def __init__(self, data_path: str = None, gpu_type: str = "A100"):
        self.data = pd.DataFrame()
        self.interpolation_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.gpu_type = gpu_type
        
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
                "sm_count": 108
            },
            "H100": {
                "peak_power_w": 700,
                "memory_bandwidth_gb_s": 3350,
                "compute_throughput_tflops": 989,  # BF16
                "base_temp_c": 30,
                "thermal_throttle_temp_c": 90,
                "memory_size_gb": 80,
                "sm_count": 132
            },
            "V100": {
                "peak_power_w": 300,
                "memory_bandwidth_gb_s": 900,
                "compute_throughput_tflops": 125,  # FP16
                "base_temp_c": 30,
                "thermal_throttle_temp_c": 83,
                "memory_size_gb": 32,
                "sm_count": 80
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
        
        return {
            "energy_joules": max(total_energy_j, 1e-6),
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
        
        # Apply thermal throttling effects
        if current_temp is not None and isinstance(current_temp, (float, int)):
            thermal_factor = self._calculate_thermal_factor(float(current_temp))
            base_costs["latency_ms"] *= thermal_factor
            base_costs["energy_joules"] *= thermal_factor
        
        # Apply memory pressure effects
        if memory_pressure > 0.7:  # High memory pressure
            memory_factor = 1.0 + (memory_pressure - 0.7) * 2.0  # Up to 60% slowdown
            base_costs["latency_ms"] *= memory_factor
            base_costs["energy_joules"] *= memory_factor * 0.8  # Less energy increase than latency
        
        self.interpolation_cache[cache_key] = base_costs
        return base_costs

    def _get_base_cost(self, op_type: str, batch_size: int) -> Dict[str, float]:
        """Get base cost with interpolation logic"""
        try:
            return self.data.loc[(op_type, batch_size)].to_dict()
        except KeyError:
            # Interpolation logic (similar to original but enhanced)
            available_batches = sorted([idx[1] for idx in self.data.index if idx[0] == op_type])
            
            if not available_batches:
                return {"energy_joules": 0.001, "latency_ms": 0.1, "temp_impact": 0.001}
            
            # Find bounds
            lower = max([s for s in available_batches if s <= batch_size], default=available_batches[0])
            upper = min([s for s in available_batches if s >= batch_size], default=available_batches[-1])
            
            if lower == upper:
                return self.data.loc[(op_type, lower)].to_dict()
            
            # Logarithmic interpolation for better scaling behavior
            log_batch = math.log(batch_size)
            log_lower = math.log(lower)
            log_upper = math.log(upper)
            alpha = (log_batch - log_lower) / (log_upper - log_lower)
            
            lower_costs = self.data.loc[(op_type, lower)].to_dict()
            upper_costs = self.data.loc[(op_type, upper)].to_dict()
            
            result = {}
            for key in lower_costs:
                result[key] = lower_costs[key] * (1 - alpha) + upper_costs[key] * alpha
            
            return result

    def _calculate_thermal_factor(self, current_temp: float) -> float:
        """Calculate thermal throttling factor based on current temperature"""
        base_temp = self.gpu_specs["base_temp_c"]
        throttle_temp = self.gpu_specs["thermal_throttle_temp_c"]
        
        if current_temp <= base_temp + 10:
            return 1.0  # No throttling at normal temps
        elif current_temp >= throttle_temp:
            return 2.0  # Significant throttling at thermal limit
        else:
            # Smooth throttling curve
            temp_ratio = (current_temp - base_temp - 10) / (throttle_temp - base_temp - 10)
            return 1.0 + temp_ratio * 1.0  # Up to 100% slowdown

    def get_cost_breakdown(
        self,
        op_type: str,
        batch_size: int,
        current_temp: float = None,
        memory_pressure: float = 0.0
    ) -> Dict[str, Any]:
        """Detailed cost breakdown, with thermal and memory adjustments."""
        # Default current_temp to 0.0 if None
        if current_temp is None:
            current_temp = 0.0
        base_costs = self.get_cost(
            op_type,
            batch_size,
            current_temp=current_temp,
            memory_pressure=memory_pressure
        )

        return {
            "base_costs": base_costs,
            "bottleneck": "memory"
                if base_costs.get("memory_utilization", 0) > base_costs.get("compute_utilization", 0)
                else "compute",
            "efficiency_score": min(
                base_costs.get("compute_utilization", 0),
                base_costs.get("memory_utilization", 0)
            ),
            "power_profile": {
                "peak_power_w": base_costs["energy_joules"] / (base_costs["latency_ms"] / 1000),
                "energy_efficiency_gflops_w": (batch_size * 1000) / base_costs["energy_joules"]
            }
        }

    def get_all_op_types(self) -> List[str]:
        """Returns all supported operation types"""
        return self.unique_op_types.copy()

    def get_thermal_safe_batch_size(self, op_type: str, current_temp: float, 
                                   max_temp_increase: float = 5.0) -> int:
        """Recommend maximum batch size to stay within thermal limits"""
        if op_type is None:
            op_type = "moe_router"
        if current_temp is None:
            current_temp = 0.0
        available_batches = sorted([idx[1] for idx in self.data.index if idx[0] == op_type])
        for batch_size in reversed(available_batches):  # Start from largest
            cost = self.get_cost(op_type, batch_size, current_temp)
            if cost["temp_impact"] <= max_temp_increase:
                return batch_size
        return available_batches[0] if available_batches else 1
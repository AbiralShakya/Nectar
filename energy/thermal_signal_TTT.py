import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from energy.profiler import GPUMetrics
from energy.profiler import GPUProfiler

class ThermalState(Enum):
    """Enumeration of thermal states for the system."""
    COOL = "cool"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"
    THROTTLED = "throttled" # Added for explicit throttled state

class PowerMode(Enum):
    """Enumeration of power modes."""
    LOW_POWER = "low_power"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    EMERGENCY = "emergency"


@dataclass
class ThermalBudget:
    """Represents current thermal constraints and budgets."""
    max_temperature: float
    max_power: float
    max_energy_per_token: float # This might be dynamic based on expert costs
    current_temperature: float
    current_power: float
    thermal_headroom: float
    power_headroom: float
    recommended_experts: List[str] # Now a list for specific expert IDs
    throttle_factor: float = 1.0


@dataclass
class ThermalSignal:
    """Complete thermal signal with all relevant information."""
    timestamp: float
    thermal_state: ThermalState
    power_mode: PowerMode
    temperature: float
    power_draw: float
    thermal_budget: ThermalBudget
    expert_recommendations: Dict[str, float]  # expert_id -> priority score
    throttle_recommendations: Dict[str, float]  # operation -> throttle factor (e.g., 'global_throttle': 0.8)
    emergency_actions: List[str]


class ThermalSignalGenerator:
    """
    Generates thermal signals based on GPU profiling data and thermal budgets.
    This module determines when to throttle computation, switch experts, or
    take emergency actions based on thermal constraints.
    """

    def __init__(self,
                 profiler: GPUProfiler,
                 cost_table_path: str = "energy/cost_table.json"):
        self.profiler = profiler
        self.cost_table_path = Path(cost_table_path)
        self.cost_table = self._load_cost_table()

        # Thermal parameters from cost table
        self.thermal_params = self.cost_table.get("thermal_parameters", {})
        self.energy_budgets = self.cost_table.get("energy_budgets", {})
        self.expert_profiles = self.cost_table.get("expert_profiles", {})

        # State tracking
        self.current_mode = PowerMode.BALANCED
        self.thermal_history: List[ThermalSignal] = []
        self.last_signal_time = 0.0
        self.emergency_cooldown_active = False # Flag for emergency state
        self.emergency_cooldown_duration = self.thermal_params.get("emergency_cooldown_duration", 30.0)
        self.emergency_cooldown_start_time = 0.0

        # Thermal model parameters
        self.base_temp = self.thermal_params.get("base_temperature", 45.0)
        self.warm_temp_threshold = self.thermal_params.get("warm_temperature_threshold", self.base_temp + 20)
        self.hot_temp_threshold = self.thermal_params.get("hot_temperature_threshold", 83.0) # Renamed for consistency
        self.critical_temp = self.thermal_params.get("critical_temperature", 87.0)
        self.thermal_time_constant = self.thermal_params.get("thermal_time_constant", 15.0)

        logging.basicConfig(level=logging.INFO) # Ensure logging is configured
        logging.info(f"ThermalSignal initialized with thresholds: {self.hot_temp_threshold}°C hot, {self.critical_temp}°C critical")

    def _load_cost_table(self) -> Dict:
        """Load the cost table configuration."""
        try:
            with open(self.cost_table_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load cost table from {self.cost_table_path}: {e}")
            return self._get_default_cost_table()

    def _get_default_cost_table(self) -> Dict:
        """Return default cost table if loading fails."""
        return {
            "thermal_parameters": {
                "base_temperature": 45.0,
                "warm_temperature_threshold": 65.0, # Added specific warm threshold
                "hot_temperature_threshold": 83.0,
                "critical_temperature": 87.0,
                "thermal_time_constant": 15.0,
                "emergency_cooldown_duration": 30.0
            },
            "energy_budgets": {
                "low_power": {"max_power_watts": 200, "max_temperature": 75.0, "max_energy_per_token_mj": 5.0},
                "balanced": {"max_power_watts": 350, "max_temperature": 80.0, "max_energy_per_token_mj": 3.0},
                "performance": {"max_power_watts": 450, "max_temperature": 85.0, "max_energy_per_token_mj": 1.5}
            },
            "expert_profiles": {
                "expert_A": {"average_power_watts": 50, "energy_per_token_mj": 2.0, "thermal_impact": 0.1},
                "expert_B": {"average_power_watts": 70, "energy_per_token_mj": 2.5, "thermal_impact": 0.15},
                "expert_C": {"average_power_watts": 30, "energy_per_token_mj": 1.0, "thermal_impact": 0.05}
            }
        }

    def get_thermal_signal(self) -> Optional[ThermalSignal]:
        """
        Generate current thermal signal based on GPU metrics and thermal model.
        """
        current_metrics = self.profiler.get_current_metrics()
        if not current_metrics:
            logging.warning("No GPU metrics available to generate thermal signal.")
            return None

        current_time = time.time()

        # Check for active emergency cooldown
        if self.emergency_cooldown_active:
            if current_time - self.emergency_cooldown_start_time < self.emergency_cooldown_duration:
                # Still in emergency cooldown, prioritize cool-down actions
                logging.info(f"Emergency cooldown active. Remaining: {self.emergency_cooldown_duration - (current_time - self.emergency_cooldown_start_time):.1f}s")
                # Force emergency mode during cooldown
                power_mode = PowerMode.EMERGENCY
                thermal_state = ThermalState.THROTTLED # Or CRITICAL if temp is still high
                if current_metrics.temperature < self.warm_temp_threshold: # Exit emergency if sufficiently cooled
                    self.emergency_cooldown_active = False
                    logging.info("Exiting emergency cooldown: temperature has dropped.")
                else:
                    # Still hot during cooldown
                    if current_metrics.temperature >= self.critical_temp:
                         thermal_state = ThermalState.CRITICAL
                    elif current_metrics.temperature >= self.hot_temp_threshold:
                         thermal_state = ThermalState.HOT
                    else:
                         thermal_state = ThermalState.THROTTLED # Indicates active throttling for cooldown
            else:
                self.emergency_cooldown_active = False
                logging.info("Emergency cooldown period ended.")

        # Determine thermal state if not in active cooldown
        if not self.emergency_cooldown_active:
            thermal_state = self._classify_thermal_state(current_metrics)

        # Determine power mode (may be overridden by thermal state)
        # If emergency cooldown is active, power_mode is already set to EMERGENCY
        if not self.emergency_cooldown_active:
            power_mode = self._determine_power_mode(current_metrics, thermal_state)
        self.current_mode = power_mode # Update internal state

        # Calculate thermal budget
        thermal_budget = self._calculate_thermal_budget(current_metrics, power_mode)

        # Generate expert recommendations
        expert_recommendations = self._generate_expert_recommendations(
            current_metrics, thermal_budget, thermal_state
        )

        # Generate throttle recommendations
        throttle_recommendations = self._generate_throttle_recommendations(
            current_metrics, thermal_state, thermal_budget
        )

        # Check for emergency actions (can trigger cooldown)
        emergency_actions = self._check_emergency_actions(current_metrics, thermal_state)
        if emergency_actions and "initiate_emergency_cooldown" in emergency_actions and not self.emergency_cooldown_active:
            self.emergency_cooldown_active = True
            self.emergency_cooldown_start_time = current_time
            logging.warning("Initiating emergency cooldown due to critical thermal state!")


        signal = ThermalSignal(
            timestamp=current_time,
            thermal_state=thermal_state,
            power_mode=power_mode,
            temperature=current_metrics.temperature,
            power_draw=current_metrics.power_draw,
            thermal_budget=thermal_budget,
            expert_recommendations=expert_recommendations,
            throttle_recommendations=throttle_recommendations,
            emergency_actions=emergency_actions
        )

        # Update history
        self.thermal_history.append(signal)
        if len(self.thermal_history) > 100:  # Keep a reasonable history size
            self.thermal_history.pop(0)

        self.last_signal_time = current_time

        return signal

    def _classify_thermal_state(self, metrics: GPUMetrics) -> ThermalState:
        """Classify current thermal state based on temperature."""
        temp = metrics.temperature

        if temp >= self.critical_temp:
            return ThermalState.CRITICAL
        elif temp >= self.hot_temp_threshold:
            return ThermalState.HOT
        elif temp >= self.warm_temp_threshold:
            return ThermalState.WARM
        else:
            return ThermalState.COOL

    def _determine_power_mode(self, metrics: GPUMetrics, thermal_state: ThermalState) -> PowerMode:
        """
        Determine the appropriate power mode based on thermal state and current usage.
        This could involve hysteresis to prevent rapid mode switching.
        """
        proposed_mode = self.current_mode # Start with current mode

        # Logic for power mode transition
        if thermal_state == ThermalState.CRITICAL:
            proposed_mode = PowerMode.EMERGENCY
        elif thermal_state == ThermalState.HOT:
            if proposed_mode == PowerMode.PERFORMANCE: # Drop from performance if hot
                proposed_mode = PowerMode.BALANCED
            elif proposed_mode == PowerMode.BALANCED and metrics.power_draw > self.energy_budgets["balanced"]["max_power_watts"] * 0.9:
                # If balanced and still drawing too much power, consider low power
                proposed_mode = PowerMode.LOW_POWER
        elif thermal_state == ThermalState.WARM:
            if proposed_mode == PowerMode.PERFORMANCE and metrics.power_draw > self.energy_budgets["performance"]["max_power_watts"]:
                # If warm but drawing too much for performance, drop to balanced
                proposed_mode = PowerMode.BALANCED
            elif proposed_mode == PowerMode.LOW_POWER and metrics.temperature < self.warm_temp_threshold - 5 and metrics.gpu_utilization < 0.5:
                # If low power and cooled down, can move to balanced if not highly utilized
                proposed_mode = PowerMode.BALANCED
        elif thermal_state == ThermalState.COOL:
            if proposed_mode == PowerMode.LOW_POWER and metrics.temperature < self.base_temp + 5 and metrics.gpu_utilization < 0.3:
                proposed_mode = PowerMode.BALANCED
            elif proposed_mode == PowerMode.BALANCED and metrics.temperature < self.base_temp + 10 and metrics.gpu_utilization > 0.8:
                # If balanced and good temperature, consider performance if high utilization
                proposed_mode = PowerMode.PERFORMANCE

        # Apply hysteresis: only change mode if conditions persist or change is drastic
        # For simplicity, we'll keep it direct for now, but a real system might use a timer
        # or history to prevent oscillations.

        if proposed_mode != self.current_mode:
            logging.info(f"Power mode transition: {self.current_mode.value} -> {proposed_mode.value} due to {thermal_state.value} state.")
            self.current_mode = proposed_mode # Update internal state

        return proposed_mode

    def _calculate_thermal_budget(self, metrics: GPUMetrics, power_mode: PowerMode) -> ThermalBudget:
        """
        Calculate current thermal budget based on the selected power mode and current metrics.
        """
        mode_budget = self.energy_budgets.get(power_mode.value, self.energy_budgets["balanced"])

        max_temp = mode_budget.get("max_temperature", 80.0)
        max_power = mode_budget.get("max_power_watts", 350.0)
        max_energy_per_token = mode_budget.get("max_energy_per_token_mj", 3.0)

        thermal_headroom = max_temp - metrics.temperature
        power_headroom = max_power - metrics.power_draw

        # Simple throttle factor calculation based on temperature headroom
        throttle_factor = 1.0
        if metrics.temperature >= self.hot_temp_threshold:
            # Linear throttle as temperature approaches critical
            temp_range = self.critical_temp - self.hot_temp_threshold
            if temp_range > 0:
                throttle_factor = 1.0 - ((metrics.temperature - self.hot_temp_threshold) / temp_range) * 0.5 # Up to 50% throttle
            throttle_factor = max(0.1, throttle_factor) # Ensure it doesn't go below 10%
            logging.warning(f"Temperature is hot ({metrics.temperature}°C), applying throttle factor: {throttle_factor:.2f}")

        # Also consider power headroom for throttling
        if metrics.power_draw > max_power * 1.1: # 10% overshoot on power budget
            power_throttle = 1.0 - ((metrics.power_draw - max_power) / (max_power * 0.5)) # throttle up to 50% for 50% power overshoot
            throttle_factor = min(throttle_factor, max(0.1, power_throttle))
            logging.warning(f"Power draw ({metrics.power_draw:.1f}W) exceeds budget, adjusting throttle factor: {throttle_factor:.2f}")


        # Dummy recommended experts for now - a real system would use a model
        # or heuristics based on expert costs and current budget.
        # This part should be driven by the _generate_expert_recommendations method.
        recommended_experts: List[str] = [] # This field is populated by _generate_expert_recommendations

        return ThermalBudget(
            max_temperature=max_temp,
            max_power=max_power,
            max_energy_per_token=max_energy_per_token,
            current_temperature=metrics.temperature,
            current_power=metrics.power_draw,
            thermal_headroom=thermal_headroom,
            power_headroom=power_headroom,
            recommended_experts=recommended_experts, # Will be filled by another method
            throttle_factor=throttle_factor
        )

    def _generate_expert_recommendations(
        self, metrics: GPUMetrics, budget: ThermalBudget, thermal_state: ThermalState
    ) -> Dict[str, float]:
        """
        Generate recommendations for expert prioritization based on thermal budget
        and expert profiles. Experts with lower energy costs and thermal impact
        are prioritized when resources are constrained.
        """
        expert_priority: Dict[str, float] = {}

        for expert_id, profile in self.expert_profiles.items():
            # Calculate a basic priority score. Lower is better (more efficient).
            # This is a simplified heuristic. A more advanced system might
            # consider recent expert usage, task requirements, etc.
            energy_cost = profile.get("energy_per_token_mj", 100.0)
            thermal_impact = profile.get("thermal_impact", 1.0)
            avg_power = profile.get("average_power_watts", 100.0)

            score = 0.0

            # Prioritize experts with lower energy consumption
            score += energy_cost / budget.max_energy_per_token # Lower is better, so divide

            # Prioritize experts that are less thermally impactful
            score += thermal_impact * (metrics.temperature / self.critical_temp) * 5.0 # Higher impact at higher temps lowers priority more

            # Adjust based on power headroom: if power is tight, penalize high-power experts
            if budget.power_headroom < 50: # If power is getting tight (e.g., less than 50W headroom)
                score += (avg_power / budget.max_power) * 2.0 # Penalize high power consumption

            # If current state is hot or critical, heavily penalize experts that generate a lot of heat/power
            if thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
                score += (thermal_impact * 10) + (avg_power / budget.max_power * 5)

            expert_priority[expert_id] = score

        # Sort experts by priority score (lower score means higher priority)
        sorted_experts = sorted(expert_priority.items(), key=lambda item: item[1])

        # Convert back to dictionary, perhaps only returning top N or all with scores
        return {expert_id: score for expert_id, score in sorted_experts}

    def _generate_throttle_recommendations(
        self, metrics: GPUMetrics, thermal_state: ThermalState, budget: ThermalBudget
    ) -> Dict[str, float]:
        """
        Generate throttle recommendations for various operations.
        A global throttle factor is primarily determined by the thermal budget.
        """
        throttle_recs: Dict[str, float] = {}

        # The primary throttle factor comes from the thermal budget calculation
        global_throttle_factor = budget.throttle_factor
        throttle_recs["global_compute_throttle"] = global_throttle_factor

        # Add more specific throttles if needed, e.g., memory bandwidth throttle
        # based on memory temperature or utilization, if available in metrics.
        if metrics.memory_utilization > 0.9 and thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
            throttle_recs["memory_bandwidth_throttle"] = max(0.5, 1.0 - (metrics.memory_utilization - 0.9) * 5) # 50% throttle if 100% util
            logging.warning(f"High memory utilization and thermal state: {throttle_recs['memory_bandwidth_throttle']:.2f} memory throttle.")


        if thermal_state == ThermalState.CRITICAL:
            throttle_recs["global_compute_throttle"] = min(0.1, global_throttle_factor) # Aggressive throttle
            logging.critical("CRITICAL thermal state: Aggressive global throttle applied.")
        elif thermal_state == ThermalState.HOT:
            throttle_recs["global_compute_throttle"] = min(0.5, global_throttle_factor) # Moderate throttle
            logging.warning("HOT thermal state: Moderate global throttle applied.")
        elif thermal_state == ThermalState.THROTTLED: # Explicit throttled state
            throttle_recs["global_compute_throttle"] = min(0.2, global_throttle_factor) # More aggressive throttle
            logging.warning("THROTTLED thermal state: Global throttle applied for cooldown.")


        return throttle_recs

    def _check_emergency_actions(self, metrics: GPUMetrics, thermal_state: ThermalState) -> List[str]:
        """
        Check if any emergency actions are required, such as initiating a full cooldown.
        """
        emergency_actions: List[str] = []

        if thermal_state == ThermalState.CRITICAL:
            logging.critical(f"Temperature {metrics.temperature}°C is CRITICAL! Recommending emergency cooldown.")
            emergency_actions.append("initiate_emergency_cooldown")
            emergency_actions.append("alert_system_operator") # Example of other emergency actions

        # Add other conditions that might trigger emergency actions, e.g.,
        # persistent high power draw despite throttling, or fan failure detection.

        return emergency_actions
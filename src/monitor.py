# src/monitors.py
import time
import threading
from typing import Dict, Any, Tuple

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not found. GPU thermal/power monitoring will be simulated.")
    _PYNVML_AVAILABLE = False

class GpuSystemMonitor:
    """
    Monitors overall GPU thermal and energy states using pynvml or simulates them.
    Provides real-time signals and stores historical data for analysis.
    """
    def __init__(self, device_id: int = 0, update_interval: float = 0.1):
        self.device_id = device_id
        self.update_interval = update_interval
        self._current_temp = 0.0
        self._current_power_watt = 0.0
        self._thermal_state = "cool"
        self._gpu_utilization_percent = 0
        self._memory_utilization_percent = 0

        self.lock = threading.Lock() # Protects shared state
        self.history = [] # To store historical data for analysis (timestamp, temp, power, etc.)

        self._running = True
        self.handle = None

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                print(f"GpuSystemMonitor initialized for device {pynvml.nvmlDeviceGetName(self.handle)}")
            except pynvml.NVMLError as error:
                print(f"Error initializing NVML for device {device_id}: {error}")
                print("Falling back to simulated GPU data.")
                _PYNVML_AVAILABLE = False
        else:
            print("GpuSystemMonitor running in simulation mode.")

        self._start_background_update()

    def _start_background_update(self):
        thread = threading.Thread(target=self._update_loop, daemon=True)
        thread.start()

    def _update_loop(self):
        while self._running:
            with self.lock:
                timestamp = time.time()
                temp, power, gpu_util, mem_util = 0.0, 0.0, 0, 0

                if _PYNVML_AVAILABLE and self.handle:
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # mW to W
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        gpu_util = util.gpu
                        mem_util = util.memory
                    except pynvml.NVMLError as error:
                        print(f"Error querying NVML: {error}. Simulating data.")
                        temp, power, gpu_util, mem_util = self._simulate_telemetry()
                else:
                    temp, power, gpu_util, mem_util = self._simulate_telemetry()

                self._current_temp = temp
                self._current_power_watt = power
                self._gpu_utilization_percent = gpu_util
                self._memory_utilization_percent = mem_util

                # Determine thermal state based on current temperature/power
                if self._current_temp > 85 or self._current_power_watt > 250:
                    self._thermal_state = "critical"
                elif self._current_temp > 75:
                    self._thermal_state = "hot"
                elif self._current_temp > 60:
                    self._thermal_state = "warm"
                else:
                    self._thermal_state = "cool"

                # Store history (can be sampled less frequently to save memory)
                self.history.append({
                    "timestamp": timestamp,
                    "temperature": self._current_temp,
                    "power_watt": self._current_power_watt,
                    "gpu_utilization_percent": self._gpu_utilization_percent,
                    "memory_utilization_percent": self._memory_utilization_percent,
                    "thermal_state": self._thermal_state
                })
                # Keep history size manageable (e.g., last 1000 entries)
                if len(self.history) > 1000:
                    self.history.pop(0)

            time.sleep(self.update_interval)

    def _simulate_telemetry(self) -> Tuple[float, float, int, int]:
        """Simple simulation of temp/power/util for testing without real GPU."""
        # This simulation can be made more sophisticated to mimic real GPU behavior
        # during different load patterns.
        current_time_ms = int(time.time() * 1000) % 20000 # Cycle every 20 seconds
        temp = 40 + 25 * (abs(current_time_ms - 10000) / 10000) # Simulates a temp swing between 40-65
        power = 50 + 70 * (abs(current_time_ms - 10000) / 10000) # Simulates power swing between 50-120
        gpu_util = int(20 + 70 * (abs(current_time_ms - 10000) / 10000))
        mem_util = int(10 + 40 * (abs(current_time_ms - 10000) / 10000))
        return temp, power, gpu_util, mem_util

    def get_current_stats(self) -> Dict[str, Any]:
        """Returns the latest sampled GPU statistics."""
        with self.lock:
            return {
                "temperature": self._current_temp,
                "power_watt": self._current_power_watt,
                "thermal_state": self._thermal_state,
                "gpu_utilization_percent": self._gpu_utilization_percent,
                "memory_utilization_percent": self._memory_utilization_percent
            }

    def get_history(self) -> list:
        """Returns a copy of the collected history."""
        with self.lock:
            return self.history.copy()

    def stop(self):
        """Stops the background monitoring thread."""
        self._running = False
        if _PYNVML_AVAILABLE and self.handle:
            try:
                pynvml.nvmlShutdown()
                print("GpuSystemMonitor: NVML shutdown.")
            except pynvml.NVMLError as error:
                print(f"Error during NVML shutdown: {error}")
import os, time, threading, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import pynvml

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

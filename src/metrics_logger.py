# src/metrics_logger.py
import os
import csv
import threading
from typing import Dict, Any
from datetime import datetime

class MetricsLogger:
    """Logs various metrics to a CSV file."""
    def __init__(self, filename: str):
        self.filename = filename
        self.fieldnames = [
            "timestamp", "epoch", "batch", "workload_type", "strategy", "expert_type",
            "loss", "task_loss", "aux_loss", "energy_loss",
            "inference_latency_ms", "throughput_qps",
            "gpu_temperature_c", "gpu_power_watt", "gpu_thermal_state",
            "gpu_utilization_percent", "memory_utilization_percent", # Added
            "expert_usage_counts", # Stored as string representation of list/array
            "expert_batch_timings_ms", # Stored as string representation of dict
            "expert_cumulative_timings_ms", # Stored as string representation of dict
            "ttha_power_loss", # For Iteration 4
            "ttha_temp_loss",  # For Iteration 4
            "ttha_latency_penalty" # For Iteration 4
        ]
        self._write_header = not os.path.exists(filename)
        self.file_lock = threading.Lock() # To ensure thread-safe writing

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)


    def log(self, data: Dict[str, Any]):
        """
        Logs a single row of metrics data to the CSV file.
        """
        # Ensure all fields are present, fill missing with None or default
        row_data = {field: data.get(field, None) for field in self.fieldnames}

        # Convert complex types to string representations for CSV
        for key in ["expert_usage_counts", "expert_batch_timings_ms", "expert_cumulative_timings_ms"]:
            if row_data[key] is not None:
                row_data[key] = str(row_data[key])
        
        # Handle TTHA history
        ttha_history = data.get("ttha_history", {})
        row_data["ttha_power_loss"] = ttha_history.get("power_loss", [])[-1] if ttha_history.get("power_loss") else None
        row_data["ttha_temp_loss"] = ttha_history.get("temp_loss", [])[-1] if ttha_history.get("temp_loss") else None
        row_data["ttha_latency_penalty"] = ttha_history.get("latency_penalty", [])[-1] if ttha_history.get("latency_penalty") else None


        with self.file_lock:
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if self._write_header:
                    writer.writeheader()
                    self._write_header = False
                writer.writerow(row_data)
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
power = nvmlDeviceGetPowerUsage(handle) / 1000

class PowerLogger:
    def __init__(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

    def log(self, iteration, loss, acc):
        power = nvmlDeviceGetPowerUsage(self.handle) / 1000
        print(f"[{iteration}] Power: {power:.2f}W | Loss: {loss:.4f} | Acc: {acc:.2f}")


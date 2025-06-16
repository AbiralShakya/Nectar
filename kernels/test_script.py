from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
print("Power draw (W):", nvmlDeviceGetPowerUsage(handle) / 1000)

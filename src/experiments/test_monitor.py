from src.monitor import GpuSystemMonitor
import time

print("Testing GpuSystemMonitor...")
monitor = GpuSystemMonitor(device_id=0, update_interval=0.5)

print("\nMonitoring for 5 seconds (simulated if no pynvml):")
for _ in range(10): 
    stats = monitor.get_current_stats()
    print(f"  Current Stats: Temp={stats['temperature']:.1f}°C, Power={stats['power_watt']:.1f}W, State={stats['thermal_state']}, GPU Util={stats['gpu_utilization_percent']}%, Mem Util={stats['memory_utilization_percent']}%")
    time.sleep(0.5)

print("\nHistory (last 5 entries):")
for entry in monitor.get_history()[-5:]:
    print(f"  {entry}")

monitor.stop()
print("GpuSystemMonitor stopped.")
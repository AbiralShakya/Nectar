from src.kernelcostmodel import KernelCostModel

print("Testing KernelCostModel...")
kcm = KernelCostModel(data_path="kernel_cost_models/kernel_cost_model_d4096.json", gpu_type="A100")

print("\nTesting exact match (batch_size=32, op_type=ffn_gate):")
cost_exact = kcm.get_cost("ffn_gate", 32)
print(f"  Cost: {cost_exact}")

print("\nTesting interpolation (batch_size=100, op_type=ffn_gate, simulating current hardware state):")
cost_interp_normal = kcm.get_cost("ffn_gate", 100, current_temp=50.0, memory_pressure=0.3)
print(f"  Cost (normal): {cost_interp_normal}")

print("\nTesting thermal throttling (batch_size=100, op_type=ffn_gate, hot GPU):")
cost_interp_hot = kcm.get_cost("ffn_gate", 100, current_temp=85.0, memory_pressure=0.3)
print(f"  Cost (hot): {cost_interp_hot}")
print(f"  Latency increase factor: {cost_interp_hot['latency_ms'] / cost_interp_normal['latency_ms']:.2f}x")

print("\nTesting memory pressure (batch_size=100, op_type=ffn_gate, high memory):")
cost_interp_mem = kcm.get_cost("ffn_gate", 100, current_temp=50.0, memory_pressure=0.9)
print(f"  Cost (mem pressure): {cost_interp_mem}")
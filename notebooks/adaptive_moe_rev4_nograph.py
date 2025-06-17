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


class AdaptiveRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int, thermal_signal_generator):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.thermal_signal_generator = thermal_signal_generator

    def forward(self, gate_logits: torch.Tensor):
        priorities = self.thermal_signal_generator.get_expert_priorities()
        bias = torch.tensor([priorities.get(str(i), 0.0) for i in range(self.num_experts)],
                            device=gate_logits.device)
        biased_logits = gate_logits + bias
        topk_vals, topk_indices = torch.topk(biased_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=-1)
        return topk_indices, routing_weights


class SimpleMoELayer(nn.Module):
    def __init__(self, gate: nn.Module, experts: nn.ModuleList, top_k: int, thermal_signal_generator):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.n_experts = len(experts)
        self.top_k = top_k
        self.router = AdaptiveRouter(self.n_experts, top_k, thermal_signal_generator)
        self.expert_timings: Dict[int, float] = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        num_tokens, d_model = x.shape
        device = x.device

        gate_logits = self.gate(x)
        top_k_indices, top_k_probs = self.router(gate_logits)
        gate_probs_all = F.softmax(gate_logits, dim=-1)

        top1_indices = top_k_indices[:, 0]
        expert_mask_top1 = F.one_hot(top1_indices, num_classes=self.n_experts).float()
        tokens_per_expert = expert_mask_top1.sum(dim=0)
        avg_gate_prob = gate_probs_all.mean(dim=0)
        aux_loss = (tokens_per_expert / (num_tokens + 1e-8) * avg_gate_prob).sum() * self.n_experts

        output = torch.zeros_like(x)
        expert_usage_counts = torch.zeros(self.n_experts, device=device)
        expert_batch_timings: Dict[int, float] = {}

        for expert_id in range(self.n_experts):
            expert_tokens_mask = (top_k_indices == expert_id).any(dim=-1)
            expert_token_indices = torch.where(expert_tokens_mask)[0]

            if expert_token_indices.numel() > 0:
                expert_input = x[expert_token_indices]
                expert_weights = torch.zeros(expert_token_indices.numel(), device=device)
                for i, token_idx in enumerate(expert_token_indices):
                    pos = torch.where(top_k_indices[token_idx] == expert_id)[0]
                    if pos.numel() > 0:
                        expert_weights[i] = top_k_probs[token_idx, pos].sum()

                start = time.time()
                expert_output = self.experts[expert_id](expert_input)
                duration = (time.time() - start) * 1000.0
                expert_batch_timings[expert_id] = duration
                self.expert_timings[expert_id] = self.expert_timings.get(expert_id, 0.0) + duration

                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                output[expert_token_indices] += weighted_output
                expert_usage_counts[expert_id] = expert_token_indices.numel()

        metrics = {
            "expert_usage_current": expert_usage_counts.cpu().numpy(),
            "total_assignments": expert_usage_counts.sum().item(),
            "expert_batch_timings_ms": expert_batch_timings,
            "expert_cumulative_timings_ms": self.expert_timings,
            "top_k_indices": top_k_indices.detach().cpu()
        }

        return output, aux_loss, metrics


def compute_energy_loss(selected_expert_indices: torch.Tensor, expert_profiles: Dict[str, Dict], alpha=0.001):
    energy = 0.0
    for idx in selected_expert_indices.view(-1):
        profile = expert_profiles.get(str(int(idx.item())))
        if profile:
            energy += profile.get("energy_cost", 0.0)
    return alpha * energy

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_experts, top_k, thermal_signal_generator):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
                                 for _ in range(num_experts)])
        self.moe_layer = SimpleMoELayer(self.gate, experts, top_k, thermal_signal_generator)

    def forward(self, x):
        return self.moe_layer(x)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, d_model=64):
        self.data = torch.randn(n, d_model)
        self.targets = torch.randn(n, d_model)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]
    def __len__(self): return len(self.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--profile_dir", type=str, default="tb_logs/debug")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, batch_size, epochs = 64, 32, 3

    thermal_signal = ThermalSignalGenerator(device_id=0)
    model = MoETransformerBlock(d_model, args.num_experts, args.top_k, thermal_signal).to(device)
    dataset = DummyDataset(n=512, d_model=d_model)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=tensorboard_trace_handler(args.profile_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output, aux_loss, selected_experts = model(x)
                task_loss = criterion(output, y)
                selected_indices = selected_experts["top_k_indices"]
                energy_loss = compute_energy_loss(selected_indices, thermal_signal.expert_profiles)
                loss = task_loss + energy_loss + aux_loss
                loss.backward()
                optimizer.step()
                prof.step()
            print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")
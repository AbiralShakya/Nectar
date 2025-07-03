import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig, AutoTokenizer
from models.distilgpt2_with_moe import DistilGPT2WithMoE
from models.ttt_router import SimpleTTTRouter, EnergyAwareTTTRouter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# import kernel cost model and GPU monitor
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor

def get_dataloader(tokenizer, texts, seq_length=64, batch_size=8):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt"
    )
    input_ids = enc.input_ids
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def run_experiment(router_type, model, dataloader, device, results_dir,
                   kernel_cost_model=None, gpu_monitor=None,
                   ttt_update_every=10):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    metrics = {'loss': [], 'ttt_updates': [], 'routing_diversity': [], 'estimated_power': []}
    csv_file = os.path.join(results_dir, f"{router_type}_metrics.csv")

    for epoch in range(1):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)["logits"]
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            # update GPU stats
            if gpu_monitor:
                stats = gpu_monitor.get_current_stats()
                hw_stats = {'power': stats['power_watt'], 'temp': stats['temperature']}
            else:
                hw_stats = {'power': None, 'temp': None}

            # TTT update (for energy-aware router)
            if router_type == 'energy_aware' and batch_idx % ttt_update_every == 0 and kernel_cost_model:
                # estimate energy via get_cost
                cost = kernel_cost_model.get_cost(
                    op_type='moe_router',
                    batch_size=input_ids.size(1),
                    current_temp=hw_stats['temp'],
                    memory_pressure=stats.get('memory_utilization_percent', 0)/100.0 if gpu_monitor else 0.0
                )
                estimated_energy = cost['energy_joules']
                feedback = {
                    'hardware_stats': hw_stats,
                    'estimated_energy': estimated_energy
                }
                for layer in model.transformer.transformer.h:
                    if hasattr(layer.ffn, 'router') and hasattr(layer.ffn.router, 'ttt_update'):
                        layer.ffn.router.ttt_update(feedback)

            # Routing diversity (how many experts used)
            with torch.no_grad():
                base_out = model.transformer.transformer(input_ids=input_ids)
                hidden = base_out.last_hidden_state
                flat = hidden.view(-1, hidden.size(-1))
                router = model.transformer.transformer.h[0].ffn.router
                expert_indices, _, _ = router(flat)
                diversity = (
                    torch.bincount(
                        expert_indices[:, 0], minlength=router.num_experts
                    ) > 0
                ).float().mean().item()

            metrics['loss'].append(loss.item())
            metrics['ttt_updates'].append(
                getattr(router, 'ttt_update_count', 0)
            )
            metrics['routing_diversity'].append(diversity)
            metrics['estimated_power'].append(hw_stats['power'])

            print(
                f"[{router_type}] Batch {batch_idx+1} | Loss: {loss.item():.4f} "
                f"| Diversity: {diversity:.2f} "
                f"| EstPower: {hw_stats['power']}"
            )

    # Save CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Loss', 'TTT_Updates', 'RoutingDiversity', 'EstimatedPower'])
        for i in range(len(metrics['loss'])):
            writer.writerow([
                metrics['loss'][i],
                metrics['ttt_updates'][i],
                metrics['routing_diversity'][i],
                metrics['estimated_power'][i]
            ])

    # plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(metrics['loss'], label='Loss')
    plt.title(f'{router_type} Loss')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(metrics['routing_diversity'], label='Diversity')
    plt.title(f'{router_type} Diversity')
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(metrics['estimated_power'], label='Est Power')
    plt.title(f'{router_type} Est. Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{router_type}_plots.png'))
    plt.close()

def main():
    os.makedirs('results_distilgpt2_moe', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Energy-aware routing can save watts on large models.",
        "Test-time training adapts models to new data distributions.",
        "Dynamic routing enables flexible computation allocation.",
        "Power consumption is a key constraint in edge computing."
    ] * 20
    dataloader = get_dataloader(tokenizer, texts)
    config = AutoConfig.from_pretrained("distilgpt2", local_files_only=True)

    # instantiate cost model and monitor
    kernel_cost_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor()

    # Baseline router
    model_baseline = DistilGPT2WithMoE(config, moe_num_experts=4, moe_top_k=2)
    for layer in model_baseline.transformer.transformer.h:
        layer.ffn.set_router(SimpleTTTRouter(config.hidden_size, 4, 2))
    run_experiment('baseline', model_baseline, dataloader, device,
                   'results_distilgpt2_moe', kernel_cost_model=None,
                   gpu_monitor=gpu_monitor)

    # Energy-aware router
    model_energy = DistilGPT2WithMoE(config, moe_num_experts=4, moe_top_k=2)
    for layer in model_energy.transformer.transformer.h:
        layer.ffn.set_router(EnergyAwareTTTRouter(config.hidden_size, 4, 2))
    run_experiment('energy_aware', model_energy, dataloader, device,
                   'results_distilgpt2_moe', kernel_cost_model=kernel_cost_model,
                   gpu_monitor=gpu_monitor)

    print("All experiments complete! Check results_distilgpt2_moe for CSVs and plots.")

if __name__ == '__main__':
    main()

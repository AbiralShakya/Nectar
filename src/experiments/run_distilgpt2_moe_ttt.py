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
import datetime
import argparse

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
                   ttt_update_every=10, lambda_energy=0.001,
                   num_epochs=1, num_batches=None):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    metrics = {'loss': [], 'ttt_updates': [], 'routing_diversity': [], 'estimated_power': []}
    csv_file = os.path.join(results_dir, f"{router_type}_metrics.csv")

    for epoch in range(num_epochs):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
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
                hw_stats = {'power': stats['power_watt'], 'temp': stats['temperature'],
                            'memory_pressure': stats['memory_utilization_percent']/100.0}
            else:
                hw_stats = {'power': None, 'temp': None, 'memory_pressure': 0.0}

            # TTT update (for energy-aware router)
            if router_type == 'energy_aware' and batch_idx % ttt_update_every == 0 and kernel_cost_model:
                actual_batch_size = input_ids.size(0)  # number of sequences
                seq_len = input_ids.size(1)           # tokens per sequence

                # fetch cost breakdown for calibration
                if hasattr(kernel_cost_model, 'get_cost_breakdown'):
                    breakdown = kernel_cost_model.get_cost_breakdown(
                        op_type='moe_router', batch_size=actual_batch_size,
                        current_temp=hw_stats['temp'], memory_pressure=hw_stats['memory_pressure']
                    )
                    print(f"[Cost Breakdown] {breakdown}")

                # get energy joules for the batch
                cost = kernel_cost_model.get_cost(
                    op_type='moe_router', batch_size=actual_batch_size,
                    current_temp=hw_stats['temp'], memory_pressure=hw_stats['memory_pressure']
                )
                batch_cost_j = cost['energy_joules']
                # normalize to per-expert cost
                per_seq_cost = batch_cost_j / actual_batch_size
                per_token_cost = per_seq_cost / seq_len
                top_k = getattr(model.transformer.transformer.h[0].ffn.router, 'top_k', 2)
                per_expert_cost = per_token_cost * top_k

                # provide normalized estimate to router
                feedback = {'hardware_stats': hw_stats, 'estimated_energy': per_expert_cost}
                for layer in model.transformer.transformer.h:
                    router = getattr(layer.ffn, 'router', None)
                    if router and hasattr(router, 'ttt_update'):
                        router.ttt_update(feedback)

            # Routing diversity
            with torch.no_grad():
                base_out = model.transformer.transformer(input_ids=input_ids)
                hidden = base_out.last_hidden_state
                flat = hidden.view(-1, hidden.size(-1))
                router_layer = model.transformer.transformer.h[0].ffn.router
                expert_indices, _, _ = router_layer(flat)
                diversity = (torch.bincount(expert_indices[:, 0], minlength=router_layer.num_experts) > 0).float().mean().item()

            metrics['loss'].append(loss.item())
            metrics['ttt_updates'].append(getattr(router_layer, 'ttt_update_count', 0))
            metrics['routing_diversity'].append(diversity)
            metrics['estimated_power'].append(hw_stats['power'])

            print(f"[{router_type}] Batch {batch_idx+1} | Loss: {loss.item():.4f} "
                  f"| Diversity: {diversity:.2f} | Power: {hw_stats['power']}")

    # write CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Loss', 'TTT_Updates', 'RoutingDiversity', 'EstimatedPower'])
        for i in range(len(metrics['loss'])):
            writer.writerow([metrics['loss'][i], metrics['ttt_updates'][i],
                             metrics['routing_diversity'][i], metrics['estimated_power'][i]])

    # plotting
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(metrics['loss'], label='Loss'); plt.title(f'{router_type} Loss'); plt.legend()
    plt.subplot(1,3,2)
    plt.plot(metrics['routing_diversity'], label='Diversity'); plt.title(f'{router_type} Diversity'); plt.legend()
    plt.subplot(1,3,3)
    plt.plot(metrics['estimated_power'], label='Power'); plt.title(f'{router_type} Power'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(results_dir, f'{router_type}_plots.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_energy', type=float, default=0.001)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--moe_top_k', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_length', type=int, default=64)
    parser.add_argument('--ttt_every', type=int, default=10)
    parser.add_argument('--num_batches', type=int, default=None, help='Number of batches to run (None = all)')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to run')
    args = parser.parse_args()

    jobid = os.environ.get('SLURM_JOB_ID') or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    results_dir = f'/scratch/gpfs/as0714/hardware_efficient_ml/results/hpc_kcm_test_{jobid}'
    os.makedirs(results_dir, exist_ok=True)

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

    dataloader = get_dataloader(tokenizer, texts, seq_length=args.seq_length, batch_size=args.batch_size)
    config = AutoConfig.from_pretrained("distilgpt2", local_files_only=True)

    kernel_cost_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor()

    # Baseline
    baseline = DistilGPT2WithMoE(config, moe_num_experts=args.num_experts, moe_top_k=args.moe_top_k)
    for layer in baseline.transformer.transformer.h:
        layer.ffn.set_router(SimpleTTTRouter(config.hidden_size, args.num_experts, args.moe_top_k))
    run_experiment('baseline', baseline, dataloader, device, results_dir,
                   kernel_cost_model=None, gpu_monitor=gpu_monitor,
                   num_epochs=args.num_epochs, num_batches=args.num_batches)

    # Energy-aware
    energy = DistilGPT2WithMoE(config, moe_num_experts=args.num_experts, moe_top_k=args.moe_top_k)
    for layer in energy.transformer.transformer.h:
        layer.ffn.set_router(EnergyAwareTTTRouter(config.hidden_size, args.num_experts, args.moe_top_k, lambda_energy=args.lambda_energy))
    run_experiment('energy_aware', energy, dataloader, device, results_dir,
                   kernel_cost_model=kernel_cost_model, gpu_monitor=gpu_monitor,
                   ttt_update_every=args.ttt_every, lambda_energy=args.lambda_energy,
                   num_epochs=args.num_epochs, num_batches=args.num_batches)

    print(f"Done! Results in {results_dir}")

if __name__ == '__main__':
    main()

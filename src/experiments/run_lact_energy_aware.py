#!/usr/bin/env python3
"""
LaCT Energy-Aware MoE Experiment
Based on Zhang et al.'s Large-Chunk TTT approach for energy-efficient routing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig, AutoTokenizer
from models.distilgpt2_with_moe import DistilGPT2WithMoE
from models.ttt_router import LaCTEnergyAwareTTTRouter, SimpleTTTRouter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import datetime
import argparse
import time

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

def run_lact_experiment(router_type, model, dataloader, device, results_dir,
                        kernel_cost_model=None, gpu_monitor=None,
                        lambda_energy=0.001, chunk_size=1000,
                        num_epochs=1, num_batches=None):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    metrics = {
        'loss': [], 'ttt_updates': [], 'routing_diversity': [], 
        'estimated_power': [], 'latency_ms': [], 'chunk_updates': []
    }
    csv_file = os.path.join(results_dir, f"{router_type}_lact_metrics.csv")

    for epoch in range(num_epochs):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            start_time = time.time()
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)["logits"]
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), labels.view(-1)
            )
            loss.backward()
            optimizer.step()
            
            latency = (time.time() - start_time) * 1000  # Convert to ms

            # update GPU stats
            if gpu_monitor:
                stats = gpu_monitor.get_current_stats()
                hw_stats = {'power': stats['power_watt'], 'temp': stats['temperature'],
                            'memory_pressure': stats['memory_utilization_percent']/100.0}
            else:
                hw_stats = {'power': None, 'temp': None, 'memory_pressure': 0.0}

            # LaCT TTT update (accumulates over chunk_size tokens)
            if router_type == 'lact_energy_aware' and kernel_cost_model:
                actual_batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                token_count = actual_batch_size * seq_len

                # Get energy cost
                cost = kernel_cost_model.get_cost(
                    op_type='moe_router', batch_size=actual_batch_size,
                    current_temp=hw_stats['temp'], memory_pressure=hw_stats['memory_pressure']
                )
                batch_cost_j = cost['energy_joules']
                per_seq_cost = batch_cost_j / actual_batch_size
                per_token_cost = per_seq_cost / seq_len
                top_k = getattr(model.transformer.transformer.h[0].ffn.router, 'top_k', 2)
                per_expert_cost = per_token_cost * top_k

                # Get expert usage
                expert_usage = None
                for layer in model.transformer.transformer.h:
                    router = getattr(layer.ffn, 'router', None)
                    if router and hasattr(router, 'forward'):
                        with torch.no_grad():
                            base_out = model.transformer.transformer(input_ids=input_ids)
                            hidden = base_out.last_hidden_state
                            flat = hidden.view(-1, hidden.size(-1))
                            _, _, metadata = router(flat)
                            if 'expert_usage' in metadata:
                                expert_usage = metadata['expert_usage']
                                break

                # LaCT feedback (accumulates over chunks)
                feedback = {
                    'hardware_stats': hw_stats, 
                    'estimated_energy': per_expert_cost,
                    'expert_usage': expert_usage,
                    'token_count': token_count,
                    'batch_size': actual_batch_size,
                    'seq_length': seq_len
                }
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
                expert_indices, _, metadata = router_layer(flat)
                diversity = (torch.bincount(expert_indices[:, 0], minlength=router_layer.num_experts) > 0).float().mean().item()

            metrics['loss'].append(loss.item())
            metrics['ttt_updates'].append(getattr(router_layer, 'ttt_update_count', 0))
            metrics['routing_diversity'].append(diversity)
            metrics['estimated_power'].append(hw_stats['power'])
            metrics['latency_ms'].append(latency)
            metrics['chunk_updates'].append(metadata.get('chunk_token_count', 0))

            print(f"[{router_type}] Batch {batch_idx+1} | Loss: {loss.item():.4f} "
                  f"| Diversity: {diversity:.2f} | Power: {hw_stats['power']:.1f}W "
                  f"| Latency: {latency:.2f}ms | Chunk: {metadata.get('chunk_token_count', 0)}")

    # write CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Loss', 'TTT_Updates', 'RoutingDiversity', 'EstimatedPower', 'Latency_ms', 'ChunkTokens'])
        for i in range(len(metrics['loss'])):
            writer.writerow([metrics['loss'][i], metrics['ttt_updates'][i],
                             metrics['routing_diversity'][i], metrics['estimated_power'][i],
                             metrics['latency_ms'][i], metrics['chunk_updates'][i]])

    # plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.plot(metrics['loss'], label='Loss')
    plt.title(f'{router_type} Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(metrics['routing_diversity'], label='Diversity')
    plt.title(f'{router_type} Diversity')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(metrics['estimated_power'], label='Power')
    plt.title(f'{router_type} Power')
    plt.legend()
    
    plt.subplot(1, 4, 4)
    plt.plot(metrics['latency_ms'], label='Latency')
    plt.title(f'{router_type} Latency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{router_type}_lact_plots.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='LaCT Energy-Aware MoE Experiment')
    parser.add_argument('--lambda_energy', type=float, default=0.001)
    parser.add_argument('--chunk_size', type=int, default=1000, help='LaCT chunk size in tokens')
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--moe_top_k', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_length', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=None, help='Number of batches to run (None = all)')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to run')
    args = parser.parse_args()

    jobid = os.environ.get('SLURM_JOB_ID') or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    results_dir = f'/scratch/gpfs/as0714/hardware_efficient_ml/results/lact_energy_test_{jobid}'
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generate more text data for large-chunk testing
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Energy-aware routing can save watts on large models.",
        "Test-time training adapts models to new data distributions.",
        "Dynamic routing enables flexible computation allocation.",
        "Power consumption is a key constraint in edge computing.",
        "Large-chunk TTT improves GPU utilization significantly.",
        "SwiGLU activation functions provide better gradient flow.",
        "Muon updates stabilize training at large chunk sizes.",
        "Load balancing across experts reduces energy consumption.",
        "Adaptive penalties improve energy-accuracy trade-offs."
    ] * 50  # More data for chunk testing

    dataloader = get_dataloader(tokenizer, texts, seq_length=args.seq_length, batch_size=args.batch_size)
    config = AutoConfig.from_pretrained("distilgpt2", local_files_only=True)

    kernel_cost_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor()

    print(f"Running LaCT Energy-Aware MoE Experiment")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Lambda energy: {args.lambda_energy}")
    print(f"Results will be saved to: {results_dir}")

    # Baseline (Simple TTT Router)
    baseline = DistilGPT2WithMoE(config, moe_num_experts=args.num_experts, moe_top_k=args.moe_top_k)
    for layer in baseline.transformer.transformer.h:
        layer.ffn.set_router(SimpleTTTRouter(config.hidden_size, args.num_experts, args.moe_top_k))
    run_lact_experiment('baseline', baseline, dataloader, device, results_dir,
                        kernel_cost_model=None, gpu_monitor=gpu_monitor,
                        num_epochs=args.num_epochs, num_batches=args.num_batches)

    # LaCT Energy-Aware Router
    lact_energy = DistilGPT2WithMoE(config, moe_num_experts=args.num_experts, moe_top_k=args.moe_top_k)
    for layer in lact_energy.transformer.transformer.h:
        layer.ffn.set_router(LaCTEnergyAwareTTTRouter(
            config.hidden_size, args.num_experts, args.moe_top_k, 
            lambda_energy=args.lambda_energy, chunk_size=args.chunk_size
        ))
    run_lact_experiment('lact_energy_aware', lact_energy, dataloader, device, results_dir,
                        kernel_cost_model=kernel_cost_model, gpu_monitor=gpu_monitor,
                        lambda_energy=args.lambda_energy, chunk_size=args.chunk_size,
                        num_epochs=args.num_epochs, num_batches=args.num_batches)

    print(f"Done! Results in {results_dir}")

if __name__ == '__main__':
    main() 
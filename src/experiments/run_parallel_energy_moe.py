#!/usr/bin/env python3
"""
Comprehensive Parallel Energy-Aware MoE Experiment Runner

This script runs the full parallel energy-aware MoE system with:
1. Dynamic expert rerouting based on batch distribution patterns
2. Energy optimization (joules per token)
3. Multi-GPU parallel execution
4. Test-Time Training (TTT) adaptation
5. SLURM integration support
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import argparse
import time
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import asdict

# Import our parallel MoE system
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.parallel_moe_system import (
    ParallelMoEConfig, ParallelMoELayer, create_parallel_moe_system,
    setup_distributed, cleanup_distributed
)
from src.moe_models import MoEConfig
from src.monitor import GpuSystemMonitor
from src.kernelcostmodel import KernelCostModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDataset:
    """Synthetic dataset for MoE experiments"""
    
    def __init__(self, num_samples: int, seq_length: int, d_model: int, vocab_size: int = 50000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Generate synthetic token sequences
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
        
        # Generate synthetic embeddings (simulating embedded tokens)
        self.embeddings = torch.randn(num_samples, seq_length, d_model)
        
        # Generate synthetic labels for language modeling
        self.labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx]
        }

class ParallelMoEModel(nn.Module):
    """Complete MoE model with parallel energy-aware layer"""
    
    def __init__(self, config: ParallelMoEConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(50000, config.moe_config.d_model)
        
        # Parallel MoE layer
        self.moe_layer = create_parallel_moe_system(config)
        
        # Output projection
        self.output_proj = nn.Linear(config.moe_config.d_model, 50000)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    async def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass with async MoE execution"""
        # Embedding
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # MoE layer (async)
        x = await self.moe_layer(x)
        
        # Output projection
        logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            # Reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = self.criterion(logits_flat, labels_flat)
        
        return {'logits': logits, 'loss': loss}

class EnergyAwareMoETrainer:
    """Trainer for energy-aware parallel MoE system"""
    
    def __init__(self, 
                 model: ParallelMoEModel,
                 config: ParallelMoEConfig,
                 device_ids: List[int],
                 output_dir: str):
        self.model = model
        self.config = config
        self.device_ids = device_ids
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        
        # Performance tracking
        self.training_metrics = []
        self.energy_metrics = []
        self.routing_metrics = []
        
        # GPU monitors for all devices
        self.gpu_monitors = {
            device_id: GpuSystemMonitor(device_id) 
            for device_id in device_ids
        }
        
        logger.info(f"Initialized EnergyAwareMoETrainer with {len(device_ids)} devices")
    
    async def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch with energy tracking"""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device_ids[0])
            labels = batch['labels'].to(self.device_ids[0])
            
            # Get hardware states before forward pass
            hardware_states_before = self._get_all_hardware_states()
            
            # Forward pass (async)
            self.optimizer.zero_grad()
            outputs = await self.model(input_ids, labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Get hardware states after forward pass
            hardware_states_after = self._get_all_hardware_states()
            
            # Calculate energy consumption
            batch_energy_metrics = self._calculate_batch_energy_metrics(
                hardware_states_before, hardware_states_after, batch_start_time
            )
            
            # Get MoE performance stats
            moe_stats = self.model.moe_layer.get_performance_stats()
            
            # Update tracking
            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            
            total_loss += loss.item()
            total_tokens += num_tokens
            batch_count += 1
            
            # Log batch metrics
            batch_metrics = {
                'epoch': epoch,
                'batch': batch_idx,
                'loss': loss.item(),
                'tokens': num_tokens,
                'batch_time_ms': (time.time() - batch_start_time) * 1000,
                'learning_rate': self.scheduler.get_last_lr()[0],
                **batch_energy_metrics,
                **moe_stats
            }
            
            self.training_metrics.append(batch_metrics)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={loss.item():.4f}, "
                    f"Power={batch_energy_metrics.get('avg_power_watts', 0):.1f}W, "
                    f"Temp={batch_energy_metrics.get('avg_temperature', 0):.1f}°C, "
                    f"J/token={batch_energy_metrics.get('joules_per_token', 0):.6f}"
                )
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / batch_count
        tokens_per_second = total_tokens / epoch_time
        
        epoch_metrics = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'epoch_time_seconds': epoch_time,
            'tokens_per_second': tokens_per_second,
            'batches': batch_count
        }
        
        logger.info(
            f"Epoch {epoch} completed: "
            f"Avg Loss={avg_loss:.4f}, "
            f"Tokens/sec={tokens_per_second:.1f}, "
            f"Time={epoch_time:.1f}s"
        )
        
        return epoch_metrics
    
    def _get_all_hardware_states(self) -> Dict[int, Dict[str, Any]]:
        """Get hardware states from all GPU monitors"""
        states = {}
        for device_id, monitor in self.gpu_monitors.items():
            states[device_id] = monitor.get_current_stats()
        return states
    
    def _calculate_batch_energy_metrics(self, 
                                      states_before: Dict[int, Dict[str, Any]],
                                      states_after: Dict[int, Dict[str, Any]],
                                      batch_start_time: float) -> Dict[str, float]:
        """Calculate energy metrics for a batch"""
        batch_time = time.time() - batch_start_time
        
        # Calculate average power consumption during batch
        total_power_before = sum(state['power_watt'] for state in states_before.values())
        total_power_after = sum(state['power_watt'] for state in states_after.values())
        avg_power = (total_power_before + total_power_after) / 2
        
        # Calculate average temperature
        avg_temp_before = np.mean([state['temperature'] for state in states_before.values()])
        avg_temp_after = np.mean([state['temperature'] for state in states_after.values()])
        avg_temperature = (avg_temp_before + avg_temp_after) / 2
        
        # Estimate energy consumption
        energy_joules = avg_power * batch_time
        
        # Estimate tokens processed (approximate)
        estimated_tokens = 1000  # This should be calculated from actual batch
        joules_per_token = energy_joules / estimated_tokens if estimated_tokens > 0 else 0
        
        return {
            'avg_power_watts': avg_power,
            'avg_temperature': avg_temperature,
            'energy_joules': energy_joules,
            'joules_per_token': joules_per_token,
            'batch_time_seconds': batch_time
        }
    
    def save_results(self, epoch_metrics: List[Dict[str, float]]):
        """Save training results and metrics"""
        # Save training metrics
        training_results = {
            'config': asdict(self.config),
            'epoch_metrics': epoch_metrics,
            'batch_metrics': self.training_metrics[-1000:],  # Last 1000 batches
            'final_moe_stats': self.model.moe_layer.get_performance_stats()
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Save energy analysis
        if self.training_metrics:
            energy_analysis = self._analyze_energy_efficiency()
            with open(self.output_dir / 'energy_analysis.json', 'w') as f:
                json.dump(energy_analysis, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _analyze_energy_efficiency(self) -> Dict[str, Any]:
        """Analyze energy efficiency over training"""
        if not self.training_metrics:
            return {}
        
        # Extract energy metrics
        power_values = [m.get('avg_power_watts', 0) for m in self.training_metrics if 'avg_power_watts' in m]
        temp_values = [m.get('avg_temperature', 0) for m in self.training_metrics if 'avg_temperature' in m]
        joules_per_token = [m.get('joules_per_token', 0) for m in self.training_metrics if 'joules_per_token' in m]
        
        if not power_values:
            return {'error': 'No energy data available'}
        
        analysis = {
            'power_statistics': {
                'mean_watts': np.mean(power_values),
                'std_watts': np.std(power_values),
                'min_watts': np.min(power_values),
                'max_watts': np.max(power_values)
            },
            'temperature_statistics': {
                'mean_celsius': np.mean(temp_values) if temp_values else 0,
                'std_celsius': np.std(temp_values) if temp_values else 0,
                'min_celsius': np.min(temp_values) if temp_values else 0,
                'max_celsius': np.max(temp_values) if temp_values else 0
            },
            'energy_efficiency': {
                'mean_joules_per_token': np.mean(joules_per_token) if joules_per_token else 0,
                'std_joules_per_token': np.std(joules_per_token) if joules_per_token else 0,
                'target_joules_per_token': self.config.joules_per_token_target,
                'efficiency_improvement_percent': 0  # Calculate based on baseline
            },
            'thermal_efficiency': {
                'thermal_threshold': self.config.thermal_threshold_celsius,
                'max_temp_reached': np.max(temp_values) if temp_values else 0,
                'thermal_margin_celsius': self.config.thermal_threshold_celsius - np.max(temp_values) if temp_values else 0
            }
        }
        
        return analysis

async def run_distributed_training(rank: int, world_size: int, args):
    """Run distributed training on multiple GPUs"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    try:
        # Create configuration
        moe_config = MoEConfig(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=args.top_k,
            expert_type=args.expert_type,
            batch_size=args.batch_size
        )
        
        parallel_config = ParallelMoEConfig(
            moe_config=moe_config,
            world_size=world_size,
            num_expert_parallel=args.expert_parallel,
            num_data_parallel=args.data_parallel,
            energy_budget_watts=args.energy_budget,
            thermal_threshold_celsius=args.thermal_threshold,
            joules_per_token_target=args.joules_per_token_target,
            rerouting_enabled=args.enable_rerouting,
            ttt_enabled=args.enable_ttt,
            ttt_chunk_size=args.ttt_chunk_size,
            async_expert_execution=args.async_execution,
            mixed_precision=args.mixed_precision
        )
        
        # Create model
        model = ParallelMoEModel(parallel_config)
        
        # Create dataset
        dataset = SyntheticDataset(
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            d_model=args.d_model
        )
        
        # Create distributed sampler and dataloader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Create trainer
        device_ids = [rank]  # Each process handles one GPU
        trainer = EnergyAwareMoETrainer(
            model, parallel_config, device_ids, 
            f"{args.output_dir}/rank_{rank}"
        )
        
        # Training loop
        epoch_metrics = []
        for epoch in range(args.num_epochs):
            sampler.set_epoch(epoch)  # Important for distributed training
            
            epoch_result = await trainer.train_epoch(dataloader, epoch)
            epoch_metrics.append(epoch_result)
            
            # Synchronize across processes
            dist.barrier()
        
        # Save results (only rank 0)
        if rank == 0:
            trainer.save_results(epoch_metrics)
            
            # Print final summary
            final_stats = model.moe_layer.get_performance_stats()
            logger.info("=== Final Training Summary ===")
            logger.info(f"Energy Efficiency: {final_stats.get('joules_per_token', 0):.6f} J/token")
            logger.info(f"Average Power: {final_stats.get('avg_power_watts', 0):.1f} W")
            logger.info(f"Average Temperature: {final_stats.get('avg_temperature_celsius', 0):.1f} °C")
            logger.info(f"Energy Improvement: {final_stats.get('energy_efficiency_improvement', 0):.1f}%")
            logger.info(f"Thermal Efficiency: {final_stats.get('thermal_efficiency', 0):.1f}%")
    
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Parallel Energy-Aware MoE Training')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k experts to use')
    parser.add_argument('--expert_type', type=str, default='swiglu', 
                       choices=['swiglu', 'quantized', 'lact'], help='Expert type')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    
    # Parallelization configuration
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--expert_parallel', type=int, default=1, help='Expert parallelism degree')
    parser.add_argument('--data_parallel', type=int, default=1, help='Data parallelism degree')
    
    # Energy optimization configuration
    parser.add_argument('--energy_budget', type=float, default=400.0, help='Energy budget in watts')
    parser.add_argument('--thermal_threshold', type=float, default=80.0, help='Thermal threshold in Celsius')
    parser.add_argument('--joules_per_token_target', type=float, default=0.002, help='Target joules per token')
    
    # Feature flags
    parser.add_argument('--enable_rerouting', action='store_true', help='Enable dynamic expert rerouting')
    parser.add_argument('--enable_ttt', action='store_true', help='Enable test-time training')
    parser.add_argument('--async_execution', action='store_true', help='Enable async expert execution')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    
    # TTT configuration
    parser.add_argument('--ttt_chunk_size', type=int, default=2048, help='TTT chunk size')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='results/parallel_energy_moe', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Auto-detect world size if not specified
    if args.world_size is None:
        args.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    logger.info(f"Starting parallel energy-aware MoE training with {args.world_size} GPUs")
    logger.info(f"Configuration: {vars(args)}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(Path(args.output_dir) / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    if args.world_size > 1:
        # Multi-GPU distributed training
        mp.spawn(
            run_distributed_training,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Single GPU training
        asyncio.run(run_distributed_training(0, 1, args))

if __name__ == '__main__':
    main()
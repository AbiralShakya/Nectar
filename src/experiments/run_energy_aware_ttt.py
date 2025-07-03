#!/usr/bin/env python3
"""
Energy-Aware Test-Time Training Router Experiment

This script demonstrates the comprehensive EnergyAwareTTTRouter that combines:
1. TTT feedback extraction from transformer gradients
2. Kernel-level energy profiles and statistical load balancing  
3. Dynamic scaling based on real-time hardware state
4. Energy-aware loss functions during TTT
5. Integration with existing hardware monitoring

Inspired by "Test-Time Training Done Right" paper's large-chunk TTT principles.
"""

import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset

# Import our custom modules
from src.moe_models import MoEConfig, MoETransformerBlock, CapacityBasedRouter
from src.routers import EnergyAwareTTTRouter, RoutingStrategy
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.metrics_logger import MetricsLogger
from src.data_utils import DataLoaderManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTTFeedbackExtractor:
    """
    Extracts TTT feedback from transformer gradients and activations.
    This implements the energy-aware TTT feedback extraction component.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_hooks = []
        self.activation_hooks = []
        self.gradients = []
        self.activations = []
        
    def register_hooks(self):
        """Register hooks to capture gradients and activations."""
        def gradient_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients.append(grad_output[0].detach())
        
        def activation_hook(module, input, output):
            self.activations.append(output.detach())
        
        # Register hooks on key layers (MoE experts, attention layers)
        for name, module in self.model.named_modules():
            if 'moe_layer' in name or 'expert' in name or 'attention' in name:
                self.gradient_hooks.append(module.register_backward_hook(gradient_hook))
                self.activation_hooks.append(module.register_forward_hook(activation_hook))
    
    def extract_feedback(self) -> Dict[str, torch.Tensor]:
        """Extract TTT feedback from captured gradients and activations."""
        feedback = {}
        
        if self.gradients:
            # Compute gradient statistics for energy-aware routing
            grad_norms = torch.stack([g.norm() for g in self.gradients if g is not None])
            feedback['gradient_norms'] = grad_norms
            feedback['gradient_mean'] = grad_norms.mean()
            feedback['gradient_std'] = grad_norms.std()
            
            # Clear gradients for next iteration
            self.gradients.clear()
        
        if self.activations:
            # Compute activation energy indicators
            act_energy = torch.stack([torch.sum(a ** 2) for a in self.activations if a is not None])
            feedback['activation_energy'] = act_energy
            feedback['activation_energy_mean'] = act_energy.mean()
            
            # Clear activations for next iteration
            self.activations.clear()
        
        return feedback
    
    def cleanup(self):
        """Remove all registered hooks."""
        for hook in self.gradient_hooks:
            hook.remove()
        for hook in self.activation_hooks:
            hook.remove()
        self.gradient_hooks.clear()
        self.activation_hooks.clear()


class EnergyAwareTTTExperiment:
    """
    Comprehensive experiment for testing the EnergyAwareTTTRouter.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.moe_config = self._create_moe_config()
        self.kernel_cost_model = KernelCostModel(
            data_path=args.kernel_cost_model_json, 
            gpu_type=args.gpu_type
        )
        self.gpu_monitor = GpuSystemMonitor(device_id=args.device_id)
        self.metrics_logger = MetricsLogger(args.log_file)
        self.dataloader_manager = DataLoaderManager(d_model=args.d_model)
        
        # Create model with EnergyAwareTTTRouter
        self.model = self._create_model()
        self.ttt_feedback_extractor = TTTFeedbackExtractor(self.model)
        
        # Performance tracking
        self.energy_savings = []
        self.thermal_improvements = []
        self.ttt_update_counts = []
        
        logger.info(f"Initialized EnergyAwareTTTExperiment on device: {self.device}")
    
    def _create_moe_config(self) -> MoEConfig:
        """Create MoE configuration for the experiment."""
        return MoEConfig(
            d_model=self.args.d_model,
            num_experts=self.args.num_experts,
            top_k=self.args.top_k,
            dropout=0.1,
            use_bias=False,
            activation="swiglu",
            expert_dropout=0.0,
            use_grouped_gemm=True,
            load_balance_weight=0.01,
            router_z_loss_weight=0.001,
            capacity_factor=1.25,
            expert_type=self.args.expert_type,
            # TTT-specific parameters
            lact_fast_weight_dim_ratio=0.25,  # Fast weight dim as fraction of d_model
            lact_chunk_size=self.args.ttt_chunk_size,
            lact_update_frequency_tokens=self.args.ttt_update_frequency,
            lact_lr=self.args.energy_aware_lr
        )
    
    def _create_model(self) -> MoETransformerBlock:
        """Create model with EnergyAwareTTTRouter."""
        # Create base model
        model = MoETransformerBlock(
            self.moe_config, 
            self.kernel_cost_model, 
            self.gpu_monitor
        ).to(self.device)
        
        # Replace router with EnergyAwareTTTRouter
        model.moe_layer.router = EnergyAwareTTTRouter(
            config=self.moe_config,
            kernel_cost_model=self.kernel_cost_model,
            gpu_system_monitor=self.gpu_monitor,
            ttt_chunk_size=self.args.ttt_chunk_size,
            ttt_update_frequency=self.args.ttt_update_frequency,
            energy_aware_lr=self.args.energy_aware_lr,
            muon_enabled=self.args.muon_enabled
        ).to(self.device)
        
        return model
    
    def run_experiment(self):
        """Run the comprehensive energy-aware TTT experiment."""
        logger.info("Starting Energy-Aware TTT Experiment")
        
        # Register TTT feedback hooks
        self.ttt_feedback_extractor.register_hooks()
        
        # Get data loader
        dataloader = self.dataloader_manager.get_dataloader(
            batch_size=self.args.batch_size,
            seq_length=self.args.seq_length
        )
        
        # Training loop
        self.model.train()
        total_steps = 0
        
        for epoch in range(self.args.num_epochs):
            epoch_start_time = time.time()
            epoch_energy_savings = []
            epoch_thermal_improvements = []
            
            for batch_idx, batch in enumerate(dataloader):
                batch_start_time = time.time()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with TTT feedback extraction
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                
                # Extract TTT feedback
                ttt_feedback = self.ttt_feedback_extractor.extract_feedback()
                
                # Backward pass
                loss.backward()
                
                # Update model parameters
                if hasattr(self.model, 'optimizer'):
                    self.model.optimizer.step()
                    self.model.optimizer.zero_grad()
                
                # Update EnergyAwareTTTRouter with observed metrics
                observed_metrics = self.gpu_monitor.get_current_stats()
                observed_metrics['inference_latency_ms'] = (time.time() - batch_start_time) * 1000
                
                # Update energy-aware loss
                loss_components = self.model.moe_layer.router.update_energy_aware_loss(
                    observed_metrics,
                    target_power=self.args.target_power,
                    target_temp=self.args.target_temp,
                    target_latency=self.args.target_latency
                )
                
                # Log metrics
                self._log_metrics(epoch, batch_idx, loss.item(), loss_components, observed_metrics)
                
                # Track performance improvements
                if 'avg_energy_savings_watts' in loss_components:
                    epoch_energy_savings.append(loss_components['avg_energy_savings_watts'])
                if 'avg_thermal_improvements_c' in loss_components:
                    epoch_thermal_improvements.append(loss_components['avg_thermal_improvements_c'])
                
                total_steps += 1
                
                # Print progress
                if batch_idx % self.args.log_interval == 0:
                    self._print_progress(epoch, batch_idx, loss.item(), loss_components)
                
                # Early stopping
                if total_steps >= self.args.max_steps:
                    break
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_energy_saving = np.mean(epoch_energy_savings) if epoch_energy_savings else 0.0
            avg_thermal_improvement = np.mean(epoch_thermal_improvements) if epoch_thermal_improvements else 0.0
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"  Average energy savings: {avg_energy_saving:.2f}W")
            logger.info(f"  Average thermal improvement: {avg_thermal_improvement:.2f}°C")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint(epoch)
            
            if total_steps >= self.args.max_steps:
                break
        
        # Final statistics
        self._print_final_statistics()
        
        # Cleanup
        self.ttt_feedback_extractor.cleanup()
        self.gpu_monitor.cleanup()
    
    def _log_metrics(self, epoch: int, batch_idx: int, loss: float, 
                    loss_components: Dict[str, float], observed_metrics: Dict[str, float]):
        """Log comprehensive metrics."""
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss,
            'gpu_temperature_c': observed_metrics.get('temperature', 0.0),
            'gpu_power_watt': observed_metrics.get('power_watt', 0.0),
            'memory_utilization_percent': observed_metrics.get('memory_utilization_percent', 0.0),
            'gpu_utilization_percent': observed_metrics.get('gpu_utilization_percent', 0.0),
            'ttt_update_count': loss_components.get('ttt_update_count', 0),
            'power_loss': loss_components.get('power_loss', 0.0),
            'temp_loss': loss_components.get('temp_loss', 0.0),
            'latency_penalty': loss_components.get('latency_penalty', 0.0),
            'memory_penalty': loss_components.get('memory_penalty', 0.0),
            'throughput_bonus': loss_components.get('throughput_bonus', 0.0)
        }
        
        self.metrics_logger.log_metrics(metrics)
    
    def _print_progress(self, epoch: int, batch_idx: int, loss: float, 
                       loss_components: Dict[str, float]):
        """Print progress information."""
        ttt_count = loss_components.get('ttt_update_count', 0)
        power_loss = loss_components.get('power_loss', 0.0)
        temp_loss = loss_components.get('temp_loss', 0.0)
        
        logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                   f"Loss={loss:.4f}, TTT_Updates={ttt_count}, "
                   f"Power_Loss={power_loss:.4f}, Temp_Loss={temp_loss:.4f}")
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.args.checkpoint_dir) / f"energy_aware_ttt_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'router_state_dict': self.model.moe_layer.router.state_dict(),
            'moe_config': self.moe_config,
            'args': self.args
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _print_final_statistics(self):
        """Print final experiment statistics."""
        router_stats = self.model.moe_layer.router.get_statistics()
        
        logger.info("=== Final Experiment Statistics ===")
        logger.info(f"Total TTT updates: {router_stats.get('ttt_update_count', 0)}")
        logger.info(f"Fast weight dimension: {router_stats.get('fast_weight_dim', 0)}")
        logger.info(f"TTT chunk size: {router_stats.get('chunk_size', 0)}")
        logger.info(f"Muon enabled: {router_stats.get('muon_enabled', False)}")
        logger.info(f"Objective weights: {router_stats.get('objective_weights', {})}")
        
        if 'avg_energy_savings_watts' in router_stats:
            logger.info(f"Average energy savings: {router_stats['avg_energy_savings_watts']:.2f}W")
        if 'avg_thermal_improvements_c' in router_stats:
            logger.info(f"Average thermal improvements: {router_stats['avg_thermal_improvements_c']:.2f}°C")


def run_training_loop(model, router_type, optimizer, dataloader, args, results_dir):
    device = next(model.parameters()).device
    kernel_cost_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor()
    scaler = GradScaler()
    import pandas as pd
    import matplotlib.pyplot as plt
    metrics = {
        'losses': [], 'accuracies': [], 'real_power': [], 'real_temp': [], 'real_memory': [], 'savings': [],
        'routing_diversity': [], 'balance_loss': [], 'entropy_loss': [], 'kl_loss': []
    }
    csv_file = os.path.join(results_dir, f"{router_type}_metrics.csv")
    plots_dir = os.path.join(results_dir, f"{router_type}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            gpu_monitor.reset_stats()
            with autocast():
                outputs = model(input_ids)
                logits = outputs.logits  # [B, S, V]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                # Routing diversity metrics
                if hasattr(model.moe_layer, 'router'):
                    gate_head = nn.Linear(logits.size(-1), args.num_experts).to(device)
                    gate_logits = gate_head(logits) / args.router_temperature
                    with torch.no_grad():
                        expert_indices = model.moe_layer.router(gate_logits, input_ids.numel(), {})[0]
                        routing_decision = torch.clamp(expert_indices[..., 0], 0, args.num_experts - 1).long()
                        expert_usage = torch.bincount(routing_decision.flatten(), minlength=args.num_experts).float()
                        routing_diversity = (expert_usage > 0).sum().item() / args.num_experts
                        expert_probs = expert_usage / (expert_usage.sum() + 1e-8)
                        balance_loss = expert_usage.std()
                        entropy_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8))
                        uniform_probs = torch.ones_like(expert_probs) / args.num_experts
                        kl_loss = torch.sum(expert_probs * (torch.log(expert_probs + 1e-8) - torch.log(uniform_probs + 1e-8)))
                else:
                    routing_diversity = 0
                    balance_loss = 0
                    entropy_loss = 0
                    kl_loss = 0

                # For energy-aware, add diversity losses
                if router_type == 'energy_aware':
                    diversity_loss = (
                        args.lambda_balance * balance_loss +
                        args.lambda_entropy * entropy_loss +
                        args.lambda_kl * kl_loss
                    )
                    total_loss = loss + diversity_loss
                else:
                    total_loss = loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            stats = gpu_monitor.get_aggregate_stats()
            power = stats.get('avg_power_watt', 0)
            temp = stats.get('avg_temperature', 0)
            mem = stats.get('avg_memory_utilization_percent', 0)

            metrics['losses'].append(loss.item())
            metrics['accuracies'].append((logits.argmax(-1) == labels).float().mean().item())
            metrics['real_power'].append(power)
            metrics['real_temp'].append(temp)
            metrics['real_memory'].append(mem)
            metrics['routing_diversity'].append(routing_diversity)
            metrics['balance_loss'].append(balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss)
            metrics['entropy_loss'].append(entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss)
            metrics['kl_loss'].append(kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss)

            print(f"[{router_type}] Epoch {epoch+1} Batch {batch_idx+1} | Loss: {loss.item():.4f} | Power: {power:.1f}W | Temp: {temp:.1f}C | Routing Diversity: {routing_diversity:.3f}")

    # Save CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Loss', 'Accuracy', 'Power', 'Temp', 'Memory', 'RoutingDiversity', 'BalanceLoss', 'EntropyLoss', 'KLLoss'])
        for i in range(len(metrics['losses'])):
            writer.writerow([
                metrics['losses'][i], metrics['accuracies'][i], metrics['real_power'][i], metrics['real_temp'][i], metrics['real_memory'][i],
                metrics['routing_diversity'][i], metrics['balance_loss'][i], metrics['entropy_loss'][i], metrics['kl_loss'][i]
            ])

    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['losses'], label='Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(metrics['routing_diversity'], label='Routing Diversity')
    plt.title('Routing Diversity')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(pd.Series(metrics['real_power']).rolling(10, min_periods=1).mean(), label='Power (smoothed)')
    plt.title('Power')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(metrics['balance_loss'], label='Balance Loss')
    plt.plot(metrics['entropy_loss'], label='Entropy Loss')
    plt.plot(metrics['kl_loss'], label='KL Loss')
    plt.title('Diversity Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{router_type}_metrics.png'))
    plt.close()
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_length', type=int, default=64)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lambda_balance', type=float, default=0.01)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--router_temperature', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Princeton University operates the Della cluster.",
        "Energy-aware routing can save watts on large models.",
        "Machine learning models require significant computational resources.",
        "The transformer architecture revolutionized natural language processing.",
        "GPU acceleration is essential for training large neural networks.",
        "Mixture of experts can improve model efficiency and performance.",
        "Hardware-aware optimization is crucial for real-world deployment.",
        "Test-time training adapts models to new data distributions.",
        "Dynamic routing enables flexible computation allocation.",
        "Power consumption is a key constraint in edge computing.",
        "Thermal management affects GPU performance and reliability.",
        "Memory bandwidth limits the speed of neural network inference.",
        "Quantization reduces model size and improves efficiency.",
        "Attention mechanisms enable models to focus on relevant information.",
        "Gradient-based optimization drives model learning and adaptation.",
        "Neural networks learn hierarchical representations of data.",
        "Backpropagation efficiently computes gradients for optimization.",
        "Regularization techniques prevent overfitting in deep learning.",
        "Transfer learning leverages pre-trained models for new tasks."
    ] * 20
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=args.seq_length, return_tensors="pt")
    input_ids = enc.input_ids
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # MoE config
    router_config = MoEConfig(
        d_model=768,
        num_experts=args.num_experts,
        top_k=2,
        dropout=0.1,
        use_bias=False,
        activation="swiglu",
        expert_dropout=0.0,
        use_grouped_gemm=True,
        load_balance_weight=0.01,
        router_z_loss_weight=0.001,
        capacity_factor=1.25,
        expert_type="simple"
    )

    import random
    for router_type in ['baseline', 'energy_aware']:
        print(f"\n=== Running {router_type.upper()} experiment ===")
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model = AutoModelForCausalLM.from_pretrained("distilgpt2", local_files_only=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if router_type == 'baseline':
            model.moe_layer.router = CapacityBasedRouter(router_config).to(device)
        else:
            model.moe_layer.router = EnergyAwareTTTRouter(
                config=router_config,
                kernel_cost_model=KernelCostModel(),
                gpu_system_monitor=GpuSystemMonitor(),
                ttt_chunk_size=32,
                ttt_update_frequency=8,
                energy_aware_lr=1e-4,
                muon_enabled=True
            ).to(device)
        run_training_loop(model, router_type, optimizer, dataloader, args, args.results_dir)

    print("\nAll experiments complete! Check your results directory for CSVs and plots.")


if __name__ == '__main__':
    main() 
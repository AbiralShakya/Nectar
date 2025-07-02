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

# Import our custom modules
from src.moe_models import MoEConfig, MoETransformerBlock
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


def main():
    import torch
    import time
    from src.moe_models import MoEConfig
    from src.routers import EnergyAwareTTTRouter
    from src.kernelcostmodel import KernelCostModel
    from src.monitor import GpuSystemMonitor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Config matches the test
    moe_config = MoEConfig(
        d_model=768,
        num_experts=8,
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
    kernel_cost_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor()
    router = EnergyAwareTTTRouter(
        config=moe_config,
        kernel_cost_model=kernel_cost_model,
        gpu_system_monitor=gpu_monitor,
        ttt_chunk_size=32,
        ttt_update_frequency=8,
        energy_aware_lr=1e-4,
        muon_enabled=True
    ).to(device)

    batch_size = 32
    seq_length = 64
    num_epochs = 3
    num_batches = 10
    d_model = moe_config.d_model
    num_experts = moe_config.num_experts

    print("Running EnergyAwareTTTRouter experiment with synthetic data...")
    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            gate_logits = torch.randn(batch_size, seq_length, num_experts).to(device)
            num_tokens = batch_size * seq_length
            context = {
                'gradients': [torch.randn(batch_size, d_model).to(device)],
                'activations': [torch.randn(batch_size, d_model).to(device)],
                'loss': torch.tensor(2.0).to(device)
            }
            start = time.time()
            expert_indices, routing_weights, metadata = router(
                gate_logits, num_tokens, context
            )
            elapsed = (time.time() - start) * 1000
            observed_metrics = gpu_monitor.get_current_stats()
            observed_metrics['inference_latency_ms'] = elapsed
            loss_components = router.update_energy_aware_loss(observed_metrics)
            print(f"Epoch {epoch+1} Batch {batch_idx+1} | Loss: {loss_components['total_loss']:.4f} | TTT updates: {metadata['ttt_update_count']}")
    print("Experiment complete.")


if __name__ == "__main__":
    main() 
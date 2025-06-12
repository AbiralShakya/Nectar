import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Optional, List, Any
import time
import logging
import math
from pathlib import Path
import pynvml
from models.moe_transformer import MoETransformer
from models.router import TopKRouter
from energy.profiler import GPUMetrics
from energy.profiler import GPUProfiler
import numpy as np
from fairscale.nn.moe import MOELayer

class WikiText2Dataset(Dataset):
    # just a simulation for now, skeleton code
    def __init__(self, vocab_size: int = 1000, seq_len: int = 512, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        logging.info(f"Initialized WikiText2Dataset with {num_samples} samples, seq_len={seq_len}, vocab_size={vocab_size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Simulate text data: random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # For language modeling, the target is usually the next token
        labels = torch.cat([input_ids[1:], torch.tensor([0])]) # Simple shift

        return {"input_ids": input_ids, "labels": labels}

# --- Evaluation Loop ---

def evaluate_model(
    model: MoETransformer,
    dataloader: DataLoader,
    device: torch.device,
    profiler: GPUProfiler,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Performs an evaluation loop for the MoE model, logging inference time
    and GPU metrics.

    Args:
        model: The MoETransformer model.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on (e.g., 'cuda' or 'cpu').
        profiler: GPUProfiler instance for logging metrics.
        log_interval: How often to log progress and metrics.

    Returns:
        A dictionary containing average perplexity, total inference time,
        average power draw, and aggregated MoE metrics.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    inference_times_ms = []
    power_draws_watts = []
    temperatures_c = []
    gpu_utilizations_percent = []
    
    # Aggregated MoE metrics across all layers and batches
    aggregated_moe_metrics: Dict[str, List[float]] = {}
    
    start_time = time.time()

    logging.info(f"Starting evaluation on device: {device}")

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Measure inference time for the batch
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            # Forward pass with profiling enabled for MoE metrics
            model_output = model(input_ids, profile=True) # Enable profiling in model for detailed metrics
            logits = model_output['logits']
            aux_loss = model_output.get('aux_loss', torch.tensor(0.0)).item()
            metrics = model_output.get('metrics', {})

            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                batch_inference_time_ms = start_event.elapsed_time(end_event)
                inference_times_ms.append(batch_inference_time_ms)
            else:
                # Approximate time for CPU
                batch_inference_time_ms = (time.time() - start_time) * 1000 # Rough estimate
                inference_times_ms.append(batch_inference_time_ms)


            # Calculate loss (for perplexity)
            # Reshape logits and labels for CrossEntropyLoss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0) # Assuming 0 is padding/ignore

            total_loss += loss.item() * labels.numel() # Accumulate loss weighted by number of elements
            total_tokens += labels.numel()
            total_batches += 1

            # Log GPU metrics
            gpu_metrics = profiler.get_current_metrics()
            if gpu_metrics:
                power_draws_watts.append(gpu_metrics.power_draw)
                temperatures_c.append(gpu_metrics.temperature)
                gpu_utilizations_percent.append(gpu_metrics.gpu_utilization)

            # Aggregate MoE specific metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        # Convert arrays to lists for consistent aggregation
                        value = value.tolist()
                    if isinstance(value, (int, float)):
                        aggregated_moe_metrics.setdefault(key, []).append(value)
                    elif isinstance(value, list):
                        # Handle array-like metrics (e.g., expert_usage)
                        # We'll need to sum/average these appropriately later
                        # For now, just store them as lists of lists/arrays
                        aggregated_moe_metrics.setdefault(key, []).append(value)


            if (batch_idx + 1) % log_interval == 0:
                avg_batch_loss = total_loss / total_tokens if total_tokens > 0 else 0
                current_perplexity = math.exp(avg_batch_loss) if avg_batch_loss < 100 else float('inf') # Avoid overflow
                
                log_msg = (
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {avg_batch_loss:.4f} | "
                    f"Perplexity: {current_perplexity:.2f} | "
                    f"Batch Time: {batch_inference_time_ms:.2f} ms"
                )
                if gpu_metrics:
                    log_msg += (
                        f" | Power: {gpu_metrics.power_draw:.1f}W | "
                        f"Temp: {gpu_metrics.temperature:.1f}°C | "
                        f"GPU Util: {gpu_metrics.gpu_utilization:.1f}%"
                    )
                logging.info(log_msg)

    end_time = time.time()
    total_inference_duration_sec = end_time - start_time

    # Calculate overall averages
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    final_perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    avg_inference_time_ms = np.mean(inference_times_ms) if inference_times_ms else 0
    avg_power_draw_watts = np.mean(power_draws_watts) if power_draws_watts else 0
    avg_temperature_c = np.mean(temperatures_c) if temperatures_c else 0
    avg_gpu_utilization_percent = np.mean(gpu_utilizations_percent) if gpu_utilizations_percent else 0

    # Aggregate MoE metrics (e.g., average expert usage across batches)
    final_moe_metrics: Dict[str, Any] = {}
    for key, values_list in aggregated_moe_metrics.items():
        if isinstance(values_list[0], (int, float)):
            final_moe_metrics[f'avg_{key}'] = np.mean(values_list)
        elif isinstance(values_list[0], list) or isinstance(values_list[0], np.ndarray):
            # For expert usage arrays, sum them up and then normalize/average if needed
            # For now, just average the arrays
            final_moe_metrics[f'avg_{key}'] = np.mean(values_list, axis=0).tolist()


    results = {
        "final_perplexity": final_perplexity,
        "total_inference_duration_sec": total_inference_duration_sec,
        "avg_inference_time_per_batch_ms": avg_inference_time_ms,
        "avg_power_draw_watts": avg_power_draw_watts,
        "avg_temperature_c": avg_temperature_c,
        "avg_gpu_utilization_percent": avg_gpu_utilization_percent,
        "aggregated_moe_metrics": final_moe_metrics
    }

    logging.info("\n--- Evaluation Summary ---")
    logging.info(f"Final Perplexity: {final_perplexity:.2f}")
    logging.info(f"Total Inference Duration: {total_inference_duration_sec:.2f} seconds")
    logging.info(f"Average Batch Inference Time: {avg_inference_time_ms:.2f} ms")
    if avg_power_draw_watts > 0:
        logging.info(f"Average Power Draw: {avg_power_draw_watts:.1f} W")
        logging.info(f"Average Temperature: {avg_temperature_c:.1f} °C")
        logging.info(f"Average GPU Utilization: {avg_gpu_utilization_percent:.1f} %")
    logging.info("Aggregated MoE Metrics:")
    for k, v in final_moe_metrics.items():
        logging.info(f"  {k}: {v}")

    return results

# --- Main execution block ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. Initialize GPUProfiler
    profiler = GPUProfiler()
    
    # 3. Model Parameters (Adjust as needed for your specific MoE setup)
    VOCAB_SIZE = 10000 # Example vocab size
    D_MODEL = 512
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 2048
    N_EXPERTS = 8
    TOP_K = 2
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 4
    
    # Set which layers use MoE (e.g., every other layer)
    USE_MOE_LAYERS = [i % 2 == 1 for i in range(N_LAYERS)] # [False, True, False, True, False, True]

    # 4. Instantiate MoE Model
    logging.info("Initializing MoETransformer model...")
    model = MoETransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        max_seq_len=MAX_SEQ_LEN,
        use_moe_layers=USE_MOE_LAYERS
    ).to(device)
    logging.info(f"Model instantiated with {sum(USE_MOE_LAYERS)} MoE layers.")
    
    # Optional: Load a pre-trained checkpoint if you have one
    # checkpoint_path = "path/to/your/checkpoint.pth"
    # if Path(checkpoint_path).exists():
    #     logging.info(f"Loading model checkpoint from {checkpoint_path}...")
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     logging.info("Model checkpoint loaded.")
    # else:
    #     logging.warning("No model checkpoint found. Using randomly initialized weights.")


    # 5. Prepare Dataset and DataLoader (using simulated WikiText-2 for baseline)
    # For actual WikiText-2, you'd use torchtext or similar to load and preprocess.
    # Example: from torchtext.datasets import WikiText2
    # For now, we use our dummy dataset.
    logging.info("Preparing dataset...")
    eval_dataset = WikiText2Dataset(vocab_size=VOCAB_SIZE, seq_len=MAX_SEQ_LEN, num_samples=100) # Use a small number of samples for baseline
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logging.info(f"Evaluation DataLoader ready with {len(eval_dataloader)} batches.")

    # 6. Run Evaluation
    logging.info("\n--- Starting Baseline Inference Evaluation ---")
    baseline_results = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        profiler=profiler,
        log_interval=10
    )

    # 7. Final Sanity Checks and Cleanup
    logging.info("\n--- Sanity Checks ---")
    if baseline_results['final_perplexity'] < float('inf'):
        logging.info(f"Perplexity sanity check: {baseline_results['final_perplexity']:.2f} (lower is better, typically starts high for untrained models)")
    else:
        logging.warning("Perplexity is infinite. This might indicate issues like very high loss or training with random weights.")

    logging.info(f"Average Power Draw: {baseline_results['avg_power_draw_watts']:.2f} W")
    logging.info(f"Average Inference Time per Batch: {baseline_results['avg_inference_time_per_batch_ms']:.2f} ms")

    # Access detailed MoE metrics
    if 'aggregated_moe_metrics' in baseline_results:
        logging.info("\nDetailed MoE Metrics (Averaged):")
        for key, value in baseline_results['aggregated_moe_metrics'].items():
            logging.info(f"  {key}: {value}")

    profiler.shutdown()
    logging.info("Evaluation complete.")
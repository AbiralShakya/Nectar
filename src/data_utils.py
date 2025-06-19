# src/data_utils.py
import torch
from typing import Tuple

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int = 1000, d_model: int = 64, varying_complexity: bool = False):
        self.num_samples = num_samples
        self.d_model = d_model
        self.varying_complexity = varying_complexity
        self.data = []
        self.targets = []
        self._generate_data()

    def _generate_data(self):
        for i in range(self.num_samples):
            # Varying input complexity (e.g., more "active" features for certain samples)
            if self.varying_complexity and i % 5 == 0:
                input_tensor = torch.randn(self.d_model) * 2.0 # Higher variance
            else:
                input_tensor = torch.randn(self.d_model) * 1.0 # Standard variance
            self.data.append(input_tensor)
            self.targets.append(torch.randn(self.d_model))

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return self.num_samples

class DataLoaderManager:
    """Manages different workload types."""
    def __init__(self, d_model: int):
        self.d_model = d_model

    def get_workload(self, workload_type: str, batch_size: int, num_samples: int) -> torch.utils.data.DataLoader:
        """
        Returns a DataLoader for a specified workload type.
        Args:
            workload_type: "standard", "high_complexity", "small_batch", "large_batch".
            batch_size: Base batch size.
            num_samples: Total number of samples in the dataset.
        """
        effective_batch_size = batch_size
        varying_complexity = False

        if workload_type == "standard":
            pass
        elif workload_type == "high_complexity":
            varying_complexity = True
        elif workload_type == "small_batch":
            effective_batch_size = max(1, batch_size // 4) # Smaller batch size
        elif workload_type == "large_batch":
            effective_batch_size = batch_size * 2 # Larger batch size
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")
        
        dataset = DummyDataset(num_samples=num_samples, d_model=self.d_model, varying_complexity=varying_complexity)
        return torch.utils.data.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
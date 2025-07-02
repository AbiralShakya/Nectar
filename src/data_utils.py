#!/usr/bin/env python3
"""
Data utilities for TTT testing with real datasets and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Base class for text datasets."""
    
    def __init__(self, data_path: str, seq_length: int = 512, tokenizer=None):
        self.data_path = data_path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load data from file."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        raise NotImplementedError


class WikiTextDataset(TextDataset):
    """WikiText dataset for language modeling."""
    
    def __init__(self, data_path: str, seq_length: int = 512, split: str = 'train'):
        self.split = split
        super().__init__(data_path, seq_length)
    
    def load_data(self):
        """Load WikiText data."""
        if not os.path.exists(self.data_path):
            logger.warning(f"WikiText data not found at {self.data_path}")
            logger.info("Generating synthetic WikiText-like data...")
            self._generate_synthetic_data()
            return
        
        logger.info(f"Loading WikiText data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple tokenization (split by whitespace)
        tokens = text.split()
        
        # Create sequences
        self.data = []
        for i in range(0, len(tokens) - self.seq_length, self.seq_length // 2):
            sequence = tokens[i:i + self.seq_length]
            if len(sequence) == self.seq_length:
                self.data.append(sequence)
        
        logger.info(f"Loaded {len(self.data)} sequences from WikiText")
    
    def _generate_synthetic_data(self):
        """Generate synthetic WikiText-like data for testing."""
        logger.info("Generating synthetic WikiText data...")
        
        # Create a simple vocabulary
        vocab = [f"word_{i}" for i in range(1000)]
        vocab.extend(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"])
        
        # Generate synthetic sequences
        self.data = []
        for _ in range(100):  # 100 sequences
            sequence = np.random.choice(vocab, size=self.seq_length, replace=True)
            self.data.append(sequence.tolist())
        
        logger.info(f"Generated {len(self.data)} synthetic sequences")
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        # Convert to token IDs (simple hash-based tokenization)
        token_ids = [hash(token) % 10000 for token in sequence]
        
        # Create input and target
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'text': ' '.join(sequence)
        }


class CodeDataset(TextDataset):
    """Code dataset for code completion task."""
    
    def __init__(self, data_path: str, seq_length: int = 512, split: str = 'train'):
        self.split = split
        super().__init__(data_path, seq_length)
    
    def load_data(self):
        """Load code data."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Code data not found at {self.data_path}")
            logger.info("Generating synthetic code data...")
            self._generate_synthetic_code_data()
            return
        
        logger.info(f"Loading code data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            code_lines = f.readlines()
        
        # Create sequences from code
        self.data = []
        current_sequence = []
        
        for line in code_lines:
            current_sequence.extend(line.split())
            
            if len(current_sequence) >= self.seq_length:
                self.data.append(current_sequence[:self.seq_length])
                current_sequence = current_sequence[self.seq_length//2:]
        
        logger.info(f"Loaded {len(self.data)} code sequences")
    
    def _generate_synthetic_code_data(self):
        """Generate synthetic code data for testing."""
        logger.info("Generating synthetic code data...")
        
        # Python-like tokens
        keywords = ["def", "class", "if", "else", "for", "while", "return", "import", "from", "as"]
        functions = ["print", "len", "range", "append", "extend", "split", "join"]
        variables = ["x", "y", "z", "i", "j", "k", "data", "result", "value", "item"]
        operators = ["=", "+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">="]
        
        vocab = keywords + functions + variables + operators + ["(", ")", "[", "]", "{", "}", ",", ".", ":", ";"]
        
        # Generate synthetic code sequences
        self.data = []
        for _ in range(100):
            sequence = []
            for _ in range(self.seq_length):
                if np.random.random() < 0.3:  # 30% chance of keyword/function
                    token = np.random.choice(keywords + functions)
                elif np.random.random() < 0.5:  # 50% chance of variable
                    token = np.random.choice(variables)
                else:  # 20% chance of operator/punctuation
                    token = np.random.choice(operators + ["(", ")", "[", "]", "{", "}", ",", ".", ":", ";"])
                sequence.append(token)
            self.data.append(sequence)
        
        logger.info(f"Generated {len(self.data)} synthetic code sequences")
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        # Convert to token IDs
        token_ids = [hash(token) % 10000 for token in sequence]
        
        # Create input and target
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'text': ' '.join(sequence)
        }


class DataLoaderManager:
    """Manager for creating data loaders for different tasks."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def get_wikitext_loader(self, batch_size: int = 8, seq_length: int = 512, split: str = 'train'):
        """Get WikiText data loader for language modeling."""
        data_path = self.data_dir / f"wikitext_{split}.txt"
        
        # Download or generate data if not exists
        if not data_path.exists():
            self._download_wikitext(split)
        
        dataset = WikiTextDataset(str(data_path), seq_length, split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    
    def get_code_loader(self, batch_size: int = 8, seq_length: int = 512, split: str = 'train'):
        """Get code data loader for code completion."""
        data_path = self.data_dir / f"code_{split}.txt"
        
        # Download or generate data if not exists
        if not data_path.exists():
            self._download_code_data(split)
        
        dataset = CodeDataset(str(data_path), seq_length, split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    
    def _download_wikitext(self, split: str):
        """Download WikiText data (placeholder - would use actual download)."""
        logger.info(f"Downloading WikiText {split} data...")
        
        # For now, create a simple text file
        data_path = self.data_dir / f"wikitext_{split}.txt"
        
        # Generate some sample text
        sample_text = """
        The quick brown fox jumps over the lazy dog. This is a sample text for testing.
        Machine learning models require large amounts of data to perform well.
        Natural language processing is a field of artificial intelligence.
        Transformers have revolutionized the field of deep learning.
        """ * 100  # Repeat to make it longer
        
        with open(data_path, 'w') as f:
            f.write(sample_text)
        
        logger.info(f"Created sample WikiText data at {data_path}")
    
    def _download_code_data(self, split: str):
        """Download code data (placeholder - would use actual download)."""
        logger.info(f"Downloading code {split} data...")
        
        # For now, create a simple code file
        data_path = self.data_dir / f"code_{split}.txt"
        
        # Generate some sample code
        sample_code = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        def factorial(n):
            if n == 0:
                return 1
            return n * factorial(n-1)
        
        class Calculator:
            def __init__(self):
                self.result = 0
            
            def add(self, x):
                self.result += x
                return self.result
        """ * 50  # Repeat to make it longer
        
        with open(data_path, 'w') as f:
            f.write(sample_code)
        
        logger.info(f"Created sample code data at {data_path}")


class TaskManager:
    """Manager for different TTT tasks."""
    
    def __init__(self, task_type: str = 'language_modeling'):
        self.task_type = task_type
    
    def compute_loss(self, model_outputs: Dict[str, torch.Tensor], 
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        if self.task_type == 'language_modeling':
            return self._language_modeling_loss(model_outputs, targets)
        elif self.task_type == 'code_completion':
            return self._code_completion_loss(model_outputs, targets)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _language_modeling_loss(self, model_outputs: Dict[str, torch.Tensor], 
                               targets: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss."""
        logits = model_outputs['logits']
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    def _code_completion_loss(self, model_outputs: Dict[str, torch.Tensor], 
                             targets: torch.Tensor) -> torch.Tensor:
        """Compute code completion loss (same as language modeling for now)."""
        logits = model_outputs['logits']
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    def extract_ttt_feedback(self, model_outputs: Dict[str, torch.Tensor], 
                           loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract TTT feedback from model outputs and loss."""
        feedback = {}
        
        # Extract gradients (if available)
        if 'hidden_states' in model_outputs:
            hidden_states = model_outputs['hidden_states']
            feedback['activation_energy'] = torch.sum(hidden_states ** 2, dim=-1).mean()
        
        # Extract logits statistics
        if 'logits' in model_outputs:
            logits = model_outputs['logits']
            probs = F.softmax(logits, dim=-1)
            feedback['logits_entropy'] = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            feedback['logits_max_prob'] = probs.max(dim=-1)[0].mean()
        
        # Loss-based feedback
        feedback['loss_value'] = loss.item()
        
        return feedback


def create_dataloader_for_task(task_type: str, batch_size: int = 8, 
                              seq_length: int = 512, split: str = 'train'):
    """Convenience function to create data loader for a specific task."""
    manager = DataLoaderManager()
    
    if task_type == 'language_modeling':
        return manager.get_wikitext_loader(batch_size, seq_length, split)
    elif task_type == 'code_completion':
        return manager.get_code_loader(batch_size, seq_length, split)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_task_manager(task_type: str):
    """Convenience function to get task manager."""
    return TaskManager(task_type)
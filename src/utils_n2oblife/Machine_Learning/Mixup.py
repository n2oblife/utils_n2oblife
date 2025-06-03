import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Callable
import random


class ImageMixup:
    """
    Enhanced Mixup implementation specifically designed for image data
    Supports various mixing strategies and image formats
    """
    
    def __init__(self, 
                 alpha: float = 0.2, 
                 prob: float = 1.0,
                 mode: str = 'batch',
                 mix_criterion: str = 'ce'):
        """
        Initialize Image Mixup
        
        Args:
            alpha: Beta distribution parameter for lambda sampling
            prob: Probability of applying mixup (0.0 to 1.0)
            mode: Mixing mode - 'batch', 'pair', or 'cutmix'
            mix_criterion: Loss mixing criterion - 'ce' or 'smooth'
        """
        self.alpha = alpha
        self.prob = prob
        self.mode = mode.lower()
        self.mix_criterion = mix_criterion.lower()
        
        if self.mode not in ['batch', 'pair', 'cutmix']:
            raise ValueError("Mode must be 'batch', 'pair', or 'cutmix'")
            
    def _sample_lambda(self) -> float:
        """Sample mixing coefficient lambda from Beta distribution"""
        if self.alpha > 0:
            return np.random.beta(self.alpha, self.alpha)
        return 1.0
    
    def _validate_input(self, x: torch.Tensor, y: torch.Tensor):
        """Validate input tensors for image mixup"""
        if len(x.shape) < 3:
            raise ValueError(f"Expected image tensor with at least 3 dimensions, got {len(x.shape)}")
        
        if x.size(0) != y.size(0):
            raise ValueError(f"Batch size mismatch: images {x.size(0)}, labels {y.size(0)}")
        
        if x.size(0) < 2:
            raise ValueError("Batch size must be at least 2 for mixup")
    
    def batch_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Standard batch mixup - mix with random samples from the same batch
        
        Args:
            x: Image tensor [B, C, H, W] or [B, H, W, C]
            y: Label tensor [B] or [B, num_classes]
            
        Returns:
            mixed_x: Mixed images
            y_a: Original labels
            y_b: Mixed labels  
            lam: Mixing coefficient
        """
        self._validate_input(x, y)
        
        batch_size = x.size(0)
        lam = self._sample_lambda()
        
        # Generate random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix images - works for both [B,C,H,W] and [B,H,W,C] formats
        mixed_x = lam * x + (1 - lam) * x[index]
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def pair_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Pair mixup - mix adjacent pairs in the batch
        
        Args:
            x: Image tensor [B, C, H, W] or [B, H, W, C]
            y: Label tensor [B] or [B, num_classes]
            
        Returns:
            mixed_x: Mixed images
            y_a: Original labels
            y_b: Mixed labels
            lam: Mixing coefficient
        """
        self._validate_input(x, y)
        
        batch_size = x.size(0)
        lam = self._sample_lambda()
        
        # Ensure even batch size for pairing
        if batch_size % 2 != 0:
            x = x[:-1]
            y = y[:-1]
            batch_size -= 1
        
        # Create pairs: (0,1), (2,3), (4,5), ...
        idx_a = torch.arange(0, batch_size, 2, device=x.device)
        idx_b = torch.arange(1, batch_size, 2, device=x.device)
        
        # Mix pairs
        mixed_x = x.clone()
        mixed_x[idx_a] = lam * x[idx_a] + (1 - lam) * x[idx_b]
        mixed_x[idx_b] = lam * x[idx_b] + (1 - lam) * x[idx_a]
        
        y_a = y.clone()
        y_b = y.clone()
        y_b[idx_a] = y[idx_b]
        y_b[idx_b] = y[idx_a]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        CutMix - mix rectangular regions of images
        
        Args:
            x: Image tensor [B, C, H, W]
            y: Label tensor [B] or [B, num_classes]
            
        Returns:
            mixed_x: Mixed images
            y_a: Original labels
            y_b: Mixed labels
            lam: Mixing coefficient (adjusted for cut area)
        """
        self._validate_input(x, y)
        
        if len(x.shape) != 4:
            raise ValueError("CutMix requires 4D tensor [B, C, H, W]")
        
        batch_size, channels, height, width = x.shape
        lam = self._sample_lambda()
        
        # Generate random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Generate random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)
        
        # Random center point
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        # Bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Mix images
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup based on configured mode
        
        Args:
            x: Image tensor
            y: Label tensor
            
        Returns:
            mixed_x: Mixed images (or original if not applied)
            y_a: Original labels
            y_b: Mixed labels (or original if not applied)
            lam: Mixing coefficient (1.0 if not applied)
        """
        # Apply mixup with probability
        if random.random() > self.prob:
            return x, y, y, 1.0
        
        if self.mode == 'batch':
            return self.batch_mixup(x, y)
        elif self.mode == 'pair':
            return self.pair_mixup(x, y)
        elif self.mode == 'cutmix':
            return self.cutmix(x, y)
        else:
            return x, y, y, 1.0


class MixupLoss:
    """Enhanced loss function for mixup training"""
    
    def __init__(self, base_criterion: nn.Module = None, label_smoothing: float = 0.0):
        """
        Initialize Mixup Loss
        
        Args:
            base_criterion: Base loss function (default: CrossEntropyLoss)
            label_smoothing: Label smoothing factor for soft targets
        """
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.label_smoothing = label_smoothing
        
    def __call__(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Compute mixup loss
        
        Args:
            pred: Model predictions
            y_a: Original labels
            y_b: Mixed labels
            lam: Mixing coefficient
            
        Returns:
            Mixed loss
        """
        if self.label_smoothing > 0:
            return self._smooth_loss(pred, y_a, y_b, lam)
        else:
            return lam * self.base_criterion(pred, y_a) + (1 - lam) * self.base_criterion(pred, y_b)
    
    def _smooth_loss(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute loss with label smoothing"""
        log_prob = F.log_softmax(pred, dim=1)
        
        # Convert labels to one-hot if needed
        if len(y_a.shape) == 1:
            num_classes = pred.size(1)
            y_a_onehot = F.one_hot(y_a, num_classes).float()
            y_b_onehot = F.one_hot(y_b, num_classes).float()
        else:
            y_a_onehot = y_a.float()
            y_b_onehot = y_b.float()
        
        # Apply label smoothing
        y_a_smooth = y_a_onehot * (1 - self.label_smoothing) + self.label_smoothing / pred.size(1)
        y_b_smooth = y_b_onehot * (1 - self.label_smoothing) + self.label_smoothing / pred.size(1)
        
        # Compute mixed soft targets
        mixed_targets = lam * y_a_smooth + (1 - lam) * y_b_smooth
        
        return -torch.sum(mixed_targets * log_prob, dim=1).mean()


def mixup_accuracy(pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> float:
    """
    Compute accuracy for mixup training
    
    Args:
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
        
    Returns:
        Mixed accuracy
    """
    _, predicted = pred.max(1)
    
    correct_a = predicted.eq(y_a).float().sum().item()
    correct_b = predicted.eq(y_b).float().sum().item()
    
    total = y_a.size(0)
    mixed_acc = (lam * correct_a + (1 - lam) * correct_b) / total
    
    return mixed_acc


class ImageMixupTrainer:
    """Complete training framework with image mixup"""
    
    def __init__(self, 
                 model: nn.Module,
                 mixup_config: dict = None,
                 device: str = 'auto'):
        """
        Initialize Mixup Trainer
        
        Args:
            model: Neural network model
            mixup_config: Mixup configuration dictionary
            device: Training device
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Setup mixup
        mixup_config = mixup_config or {}
        self.mixup = ImageMixup(**mixup_config)
        
        # Setup loss
        loss_config = mixup_config.get('loss', {})
        self.criterion = MixupLoss(**loss_config)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Single training step with mixup
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            loss: Training loss
            accuracy: Training accuracy
        """
        x, y = x.to(self.device), y.to(self.device)
        
        # Apply mixup
        mixed_x, y_a, y_b, lam = self.mixup(x, y)
        
        # Forward pass
        pred = self.model(mixed_x)
        
        # Compute loss
        loss = self.criterion(pred, y_a, y_b, lam)
        
        # Compute accuracy
        accuracy = mixup_accuracy(pred, y_a, y_b, lam)
        
        return loss, accuracy
    
    def validate_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Validation step (no mixup)
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            loss: Validation loss
            accuracy: Validation accuracy
        """
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            pred = self.model(x)
            loss = F.cross_entropy(pred, y)
            
            _, predicted = pred.max(1)
            accuracy = predicted.eq(y).float().mean().item()
        
        return loss, accuracy


# Example usage and configuration
def create_mixup_configs():
    """Create example configurations for different mixup strategies"""
    
    configs = {
        'standard_mixup': {
            'alpha': 0.2,
            'prob': 1.0,
            'mode': 'batch',
            'loss': {'label_smoothing': 0.0}
        },
        
        'aggressive_mixup': {
            'alpha': 1.0,
            'prob': 0.8,
            'mode': 'batch',
            'loss': {'label_smoothing': 0.1}
        },
        
        'cutmix': {
            'alpha': 1.0,
            'prob': 0.5,
            'mode': 'cutmix',
            'loss': {'label_smoothing': 0.0}
        },
        
        'pair_mixup': {
            'alpha': 0.4,
            'prob': 1.0,
            'mode': 'pair',
            'loss': {'label_smoothing': 0.05}
        }
    }
    
    return configs


def example_training_loop():
    """
    Example of how to use the Image Mixup framework in training
    
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    # Pseudo-code example

    # Setup model and data
    model = YourImageModel()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create mixup trainer
    mixup_config = {
        'alpha': 0.2,
        'prob': 1.0,
        'mode': 'batch',
        'loss': {'label_smoothing': 0.1}
    }
    
    trainer = ImageMixupTrainer(model, mixup_config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Mixup training step
            loss, acc = trainer.train_step(images, labels)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                loss, acc = trainer.validate_step(images, labels)
                val_loss += loss.item()
                val_acc += acc
        
        print(f'Epoch {epoch}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc/len(val_loader):.4f}')
    """
    pass


# Advanced mixup variants
class ManifoldMixup(ImageMixup):
    """Manifold Mixup - mix in hidden representations"""
    
    def __init__(self, alpha: float = 0.2, mix_layers: list = None):
        super().__init__(alpha=alpha)
        self.mix_layers = mix_layers or ['layer2', 'layer3']
    
    def forward_hook(self, module, input, output):
        """Hook for capturing intermediate representations"""
        # Implementation would capture and mix hidden representations
        pass


class AdaptiveMixup(ImageMixup):
    """Adaptive Mixup - adjust alpha based on training progress"""
    
    def __init__(self, alpha_schedule: Callable = None):
        super().__init__()
        self.alpha_schedule = alpha_schedule or (lambda epoch: max(0.1, 0.4 - epoch * 0.01))
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int):
        """Update current epoch for alpha scheduling"""
        self.current_epoch = epoch
        self.alpha = self.alpha_schedule(epoch)


if __name__ == '__main__':
    print("Image Mixup Training Framework")
    print("=" * 40)
    
    # Show available configurations
    configs = create_mixup_configs()
    
    print("Available Mixup Configurations:")
    for name, config in configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nUsage:")
    print("1. Choose a mixup configuration")
    print("2. Create ImageMixupTrainer with your model")
    print("3. Use trainer.train_step() in your training loop")
    print("4. Use trainer.validate_step() for validation")
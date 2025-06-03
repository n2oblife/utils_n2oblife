import os
import argparse
import json
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


class KnowledgeDistillationLoss(nn.Module):
    """Generic Knowledge Distillation Loss supporting multiple modes"""
    
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0, mode: str = 'kl'):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mode = mode.lower()
        
        if self.mode not in ['kl', 'mse', 'cosine']:
            raise ValueError(f"Unsupported KD mode: {mode}. Choose from ['kl', 'mse', 'cosine']")
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, 
                true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_outputs: Raw logits from student model
            teacher_outputs: Raw logits from teacher model  
            true_labels: Ground truth labels
        """
        # Soft loss (distillation loss)
        if self.mode == 'kl':
            student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
            soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        elif self.mode == 'mse':
            student_soft = F.softmax(student_outputs / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
            soft_loss = F.mse_loss(student_soft, teacher_soft)
        elif self.mode == 'cosine':
            student_soft = F.softmax(student_outputs / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
            soft_loss = 1 - F.cosine_similarity(student_soft, teacher_soft, dim=1).mean()
        
        # Scale by temperature squared (for KL and MSE)
        if self.mode in ['kl', 'mse']:
            soft_loss = soft_loss * (self.temperature ** 2)
        
        # Hard loss (classification loss)
        hard_loss = F.cross_entropy(student_outputs, true_labels)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        
        return total_loss


class GenericKDTrainer:
    """Generic Knowledge Distillation Trainer for any model architecture"""
    
    def __init__(self, 
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'auto',
                 loss_config: Optional[Dict[str, Any]] = None,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 checkpoint_dir: str = './checkpoints',
                 progress_callback: Optional[Callable] = None):
        """
        Initialize the KD trainer
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to be trained
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use ('auto', 'cuda', 'cpu')
            loss_config: Configuration for KD loss
            optimizer_config: Configuration for optimizer
            scheduler_config: Configuration for learning rate scheduler
            checkpoint_dir: Directory to save checkpoints
            progress_callback: Optional callback for progress reporting
        """
        self.device = self._setup_device(device)
        
        # Setup models
        self.teacher_model = self._setup_model(teacher_model, 'teacher')
        self.student_model = self._setup_model(student_model, 'student')
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup loss function
        loss_config = loss_config or {}
        self.criterion = KnowledgeDistillationLoss(**loss_config)
        
        # Setup optimizer
        optimizer_config = optimizer_config or {'type': 'adam', 'lr': 0.001}
        self.optimizer = self._setup_optimizer(optimizer_config)
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler(scheduler_config) if scheduler_config else None
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress callback
        self.progress_callback = progress_callback or self._default_progress_callback
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and torch.cuda.is_available():
            cudnn.benchmark = True
        
        return device
    
    def _setup_model(self, model: nn.Module, model_type: str) -> nn.Module:
        """Setup model for training/inference"""
        model = model.to(self.device)
        
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        if model_type == 'teacher':
            model.eval()  # Teacher is always in eval mode
            for param in model.parameters():
                param.requires_grad = False
        
        return model
    
    def _setup_optimizer(self, config: Dict[str, Any]) -> optim.Optimizer:
        """Setup optimizer based on configuration"""
        optimizer_type = config.get('type', 'adam').lower()
        lr = config.get('lr', 0.001)
        weight_decay = config.get('weight_decay', 0.0)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.student_model.parameters(), 
                            lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = config.get('momentum', 0.9)
            return optim.SGD(self.student_model.parameters(), 
                           lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.student_model.parameters(), 
                             lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _setup_scheduler(self, config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        scheduler_type = config.get('type', '').lower()
        
        if scheduler_type == 'step':
            step_size = config.get('step_size', 30)
            gamma = config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = config.get('T_max', 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'plateau':
            patience = config.get('patience', 10)
            factor = config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                       patience=patience, factor=factor)
        return None
    
    def _default_progress_callback(self, epoch: int, batch_idx: int, total_batches: int, 
                                 loss: float, accuracy: float, phase: str):
        """Default progress reporting"""
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f'{phase} Epoch {epoch} [{batch_idx}/{total_batches}] '
                  f'Loss: {loss:.4f} | Acc: {accuracy:.2f}%')
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.student_model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get teacher outputs (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            
            # Forward pass through student
            self.optimizer.zero_grad()
            student_outputs = self.student_model(inputs)
            
            # Compute loss
            loss = self.criterion(student_outputs, teacher_outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Progress callback
            accuracy = 100.0 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            self.progress_callback(self.current_epoch, batch_idx, len(self.train_loader),
                                 avg_loss, accuracy, 'Train')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the student model"""
        self.student_model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Student outputs
                student_outputs = self.student_model(inputs)
                
                # Compute validation loss (only hard loss)
                loss = F.cross_entropy(student_outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Progress callback
                accuracy = 100.0 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                self.progress_callback(self.current_epoch, batch_idx, len(self.val_loader),
                                     avg_loss, accuracy, 'Val')
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(state, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(state, best_path)
            print(f'New best model saved with accuracy: {self.best_acc:.2f}%')
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        self.training_history = checkpoint.get('training_history', 
                                             {'train_loss': [], 'val_acc': [], 'val_loss': []})
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Checkpoint loaded from epoch {self.current_epoch} with best acc: {self.best_acc:.2f}%')
    
    def train(self, num_epochs: int, save_every: int = 10, resume_from: Optional[str] = None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1
        
        print(f'Starting training from epoch {start_epoch} to {num_epochs}')
        print(f'Device: {self.device}')
        print(f'Student model parameters: {sum(p.numel() for p in self.student_model.parameters()):,}')
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch}/{num_epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Check if best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  Best Val Acc: {self.best_acc:.2f}% | LR: {current_lr:.6f}')
        
        print(f'\nTraining completed! Best validation accuracy: {self.best_acc:.2f}%')
        return self.training_history


def load_teacher_model(model_class, checkpoint_path: str, device: str, **model_kwargs):
    """Generic function to load teacher model from checkpoint"""
    model = model_class(**model_kwargs)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f'Teacher model loaded from {checkpoint_path}')
    else:
        print(f'Warning: Teacher checkpoint not found at {checkpoint_path}')
    
    return model


def create_kd_config_template():
    """Create a template configuration file for knowledge distillation"""
    config = {
        "loss_config": {
            "alpha": 0.7,
            "temperature": 4.0,
            "mode": "kl"
        },
        "optimizer_config": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 1e-4
        },
        "scheduler_config": {
            "type": "cosine",
            "T_max": 200
        },
        "training_config": {
            "num_epochs": 200,
            "save_every": 20,
            "checkpoint_dir": "./checkpoints"
        },
        "model_config": {
            "teacher_checkpoint": "./pretrained/teacher_model.pth",
            "teacher_kwargs": {},
            "student_kwargs": {}
        }
    }
    
    return config


# Example usage function
def example_usage():
    """
    Example of how to use the generic KD framework
    
    # Define your models
    teacher_model = YourTeacherModel()
    student_model = YourStudentModel()
    
    # Load teacher model
    teacher_model = load_teacher_model(
        YourTeacherModel, 
        './checkpoints/teacher.pth', 
        device='cuda'
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Configure training
    loss_config = {'alpha': 0.7, 'temperature': 4.0, 'mode': 'kl'}
    optimizer_config = {'type': 'adam', 'lr': 0.001, 'weight_decay': 1e-4}
    scheduler_config = {'type': 'cosine', 'T_max': 200}
    
    # Create trainer
    trainer = GenericKDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        checkpoint_dir='./checkpoints'
    )
    
    # Train the model
    history = trainer.train(num_epochs=200, save_every=20)
    """
    pass


if __name__ == '__main__':
    # Create example configuration
    config = create_kd_config_template()
    
    # Save configuration template
    with open('kd_config_template.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Generic Knowledge Distillation framework created!")
    print("Configuration template saved as 'kd_config_template.json'")
    print("\nTo use this framework:")
    print("1. Define your teacher and student models")
    print("2. Create your data loaders")
    print("3. Configure the training parameters")
    print("4. Initialize GenericKDTrainer")
    print("5. Call trainer.train()")
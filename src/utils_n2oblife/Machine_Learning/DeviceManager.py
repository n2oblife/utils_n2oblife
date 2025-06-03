import os
import sys
import warnings
import logging
from typing import Optional, Dict, Any, Union, List
from contextlib import contextmanager
import psutil
import platform

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    import random
    RANDOM_AVAILABLE = True
except ImportError:
    RANDOM_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class DeviceManager:
    """
    Advanced device management class for machine learning training with comprehensive 
    seed management, device optimization, and system monitoring capabilities.
    
    This class handles device selection, reproducibility settings, memory management,
    and provides utilities for monitoring training resources.
    """
    
    def __init__(self, 
                 seed: int = 123456789,
                 device: Optional[str] = None,
                 mixed_precision: bool = True,
                 deterministic: bool = True,
                 benchmark: bool = False,
                 verbose: bool = True):
        """
        Initialize the DeviceManager with comprehensive training settings.
        
        Args:
            seed (int): Random seed for reproducibility (default: 123456789)
            device (str, optional): Force specific device ('cpu', 'cuda', 'mps', etc.)
            mixed_precision (bool): Enable mixed precision training (default: True)
            deterministic (bool): Enable deterministic operations for reproducibility (default: True)
            benchmark (bool): Enable cuDNN benchmark for performance (default: False)
            verbose (bool): Print device and configuration information (default: True)
        """
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.deterministic = deterministic
        self.benchmark = benchmark
        self.verbose = verbose
        
        # Initialize device information
        self.device_info = self._detect_devices()
        self.device = self._select_device(device)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Apply seed and optimizations
        self.setup_training_environment()
        
        if self.verbose:
            self.print_system_info()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the device manager."""
        logger = logging.getLogger('DeviceManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def _detect_devices(self) -> Dict[str, Any]:
        """
        Detect available devices and their capabilities.
        
        Returns:
            Dict containing device information
        """
        device_info = {
            'cpu': {'available': True, 'cores': psutil.cpu_count()},
            'cuda': {'available': False, 'devices': []},
            'mps': {'available': False},
            'system': {
                'platform': platform.system(),
                'python_version': sys.version,
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
        }
        
        if TORCH_AVAILABLE:
            # CUDA detection
            if torch.cuda.is_available():
                device_info['cuda']['available'] = True
                device_info['cuda']['device_count'] = torch.cuda.device_count()
                device_info['cuda']['devices'] = []
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device_info['cuda']['devices'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': round(props.total_memory / (1024**3), 2),
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
            
            # MPS (Apple Silicon) detection
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_info['mps']['available'] = True
        
        return device_info
    
    def _select_device(self, preferred_device: Optional[str] = None) -> torch.device:
        """
        Select the best available device for training.
        
        Args:
            preferred_device (str, optional): Preferred device string
            
        Returns:
            torch.device: Selected device
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required but not available")
        
        if preferred_device:
            try:
                device = torch.device(preferred_device)
                if self.verbose:
                    self.logger.info(f"Using preferred device: {device}")
                return device
            except Exception as e:
                self.logger.warning(f"Failed to use preferred device {preferred_device}: {e}")
        
        # Auto-select best device
        if self.device_info['cuda']['available']:
            # Select GPU with most memory
            best_gpu = max(self.device_info['cuda']['devices'], 
                          key=lambda x: x['memory_gb'])
            device = torch.device(f"cuda:{best_gpu['id']}")
        elif self.device_info['mps']['available']:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        if self.verbose:
            self.logger.info(f"Auto-selected device: {device}")
        
        return device
    
    def setup_training_environment(self) -> None:
        """
        Setup the complete training environment with seeds and optimizations.
        """
        self.set_seed()
        self.configure_torch()
        self.configure_cuda()
        self.configure_tensorflow()
        
        if self.verbose:
            self.logger.info("Training environment configured successfully")
    
    def set_seed(self) -> None:
        """
        Set random seeds for all relevant libraries to ensure reproducibility.
        """
        # Python hash seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # Python random
        if RANDOM_AVAILABLE:
            import random
            random.seed(self.seed)
        
        # NumPy
        if NUMPY_AVAILABLE:
            np.random.seed(self.seed)
        
        # PyTorch
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        
        # TensorFlow
        if TF_AVAILABLE:
            try:
                # TensorFlow 2.x
                tf.random.set_seed(self.seed)
            except AttributeError:
                try:
                    # TensorFlow 1.x
                    tf.set_random_seed(self.seed)
                except AttributeError:
                    pass
        
        if self.verbose:
            self.logger.info(f"Random seed set to {self.seed} for all libraries")
    
    def configure_torch(self) -> None:
        """Configure PyTorch-specific settings."""
        if not TORCH_AVAILABLE:
            return
        
        # Deterministic operations
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Thread settings for reproducibility
        torch.set_num_threads(1)
    
    def configure_cuda(self) -> None:
        """Configure CUDA-specific settings."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        # cuDNN settings
        torch.backends.cudnn.deterministic = self.deterministic
        torch.backends.cudnn.benchmark = self.benchmark
        
        # Clear cache
        torch.cuda.empty_cache()
        
        if self.verbose:
            self.logger.info(f"CUDA configured - deterministic: {self.deterministic}, benchmark: {self.benchmark}")
    
    def configure_tensorflow(self) -> None:
        """Configure TensorFlow settings if available."""
        if not TF_AVAILABLE:
            return
        
        try:
            # TensorFlow 2.x configuration
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if self.verbose:
                    self.logger.info(f"TensorFlow configured with {len(gpus)} GPUs")
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"TensorFlow configuration failed: {e}")
    
    def get_device_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information for the selected device.
        
        Returns:
            Dict containing memory information
        """
        info = {'device': str(self.device)}
        
        if self.device.type == 'cuda' and TORCH_AVAILABLE:
            device_id = self.device.index or 0
            info.update({
                'allocated_gb': round(torch.cuda.memory_allocated(device_id) / (1024**3), 2),
                'reserved_gb': round(torch.cuda.memory_reserved(device_id) / (1024**3), 2),
                'total_gb': self.device_info['cuda']['devices'][device_id]['memory_gb']
            })
        elif self.device.type == 'cpu':
            memory = psutil.virtual_memory()
            info.update({
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'total_gb': round(memory.total / (1024**3), 2),
                'percent': memory.percent
            })
        
        return info
    
    def clear_cache(self) -> None:
        """Clear device cache to free up memory."""
        if TORCH_AVAILABLE:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                if self.verbose:
                    self.logger.info("CUDA cache cleared")
            elif hasattr(torch.backends, 'mps') and self.device.type == 'mps':
                torch.mps.empty_cache()
                if self.verbose:
                    self.logger.info("MPS cache cleared")
    
    def optimize_for_inference(self) -> None:
        """Optimize settings for inference (vs training)."""
        if TORCH_AVAILABLE:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if self.verbose:
                self.logger.info("Optimized for inference")
    
    def optimize_for_training(self) -> None:
        """Optimize settings for training."""
        if TORCH_AVAILABLE:
            torch.backends.cudnn.benchmark = self.benchmark
            torch.backends.cudnn.deterministic = self.deterministic
            if self.verbose:
                self.logger.info("Optimized for training")
    
    @contextmanager
    def inference_mode(self):
        """Context manager for inference mode with optimized settings."""
        original_benchmark = getattr(torch.backends.cudnn, 'benchmark', False) if TORCH_AVAILABLE else False
        original_deterministic = getattr(torch.backends.cudnn, 'deterministic', True) if TORCH_AVAILABLE else True
        
        try:
            self.optimize_for_inference()
            yield
        finally:
            if TORCH_AVAILABLE:
                torch.backends.cudnn.benchmark = original_benchmark
                torch.backends.cudnn.deterministic = original_deterministic
    
    def move_to_device(self, obj: Union[torch.Tensor, nn.Module, Dict, List]) -> Any:
        """
        Move tensors/models to the selected device.
        
        Args:
            obj: Object to move (tensor, model, dict, or list)
            
        Returns:
            Object moved to device
        """
        if not TORCH_AVAILABLE:
            return obj
        
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, nn.Module):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item) for item in obj)
        else:
            return obj
    
    def get_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get model size information.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict with model size information
        """
        if not TORCH_AVAILABLE:
            return {}
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory_mb = param_count * 4 / (1024**2)  # Assuming float32
        
        return {
            'total_parameters': param_count,
            'trainable_parameters': trainable_count,
            'non_trainable_parameters': param_count - trainable_count,
            'estimated_memory_mb': round(param_memory_mb, 2)
        }
    
    def print_system_info(self) -> None:
        """Print comprehensive system and device information."""
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        
        # System info
        sys_info = self.device_info['system']
        print(f"Platform: {sys_info['platform']}")
        print(f"Python: {sys_info['python_version'].split()[0]}")
        print(f"System Memory: {sys_info['memory_gb']} GB")
        print(f"CPU Cores: {self.device_info['cpu']['cores']}")
        
        # PyTorch info
        if TORCH_AVAILABLE:
            print(f"PyTorch: {torch.__version__}")
        
        # Device info
        print(f"\nSelected Device: {self.device}")
        
        if self.device_info['cuda']['available']:
            print(f"CUDA Devices: {self.device_info['cuda']['device_count']}")
            for device in self.device_info['cuda']['devices']:
                print(f"  GPU {device['id']}: {device['name']} ({device['memory_gb']} GB)")
        
        if self.device_info['mps']['available']:
            print("Apple MPS: Available")
        
        # Configuration
        print(f"\nConfiguration:")
        print(f"  Seed: {self.seed}")
        print(f"  Mixed Precision: {self.mixed_precision}")
        print(f"  Deterministic: {self.deterministic}")
        print(f"  Benchmark: {self.benchmark}")
        
        print("=" * 60)
    
    def benchmark_device(self, size: tuple = (1000, 1000), iterations: int = 100) -> Dict[str, float]:
        """
        Run a simple benchmark on the selected device.
        
        Args:
            size: Size of tensors for benchmark
            iterations: Number of iterations
            
        Returns:
            Dict with benchmark results
        """
        if not TORCH_AVAILABLE:
            return {}
        
        import time
        
        # Create test tensors
        a = torch.randn(size, device=self.device)
        b = torch.randn(size, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        ops_per_sec = 1.0 / avg_time
        
        return {
            'device': str(self.device),
            'tensor_size': size,
            'iterations': iterations,
            'avg_time_ms': round(avg_time * 1000, 3),
            'ops_per_second': round(ops_per_sec, 2)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.clear_cache()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DeviceManager(device={self.device}, seed={self.seed})"
    
    
if __name__ == "__main__":
    # Basic usage (drop-in replacement)
    device_manager = DeviceManager(seed=42, verbose=True)
    model = model.to(device_manager.device)

    # Advanced configuration
    dm = DeviceManager(
        seed=123,
        device="cuda:1",  # Force specific GPU
        mixed_precision=True,
        deterministic=True,
        benchmark=False
    )

    # Context manager with cleanup
    with DeviceManager(seed=42) as dm:
        model = dm.move_to_device(model)
        data = dm.move_to_device(data)
        # Training code here...
        # Automatic cleanup on exit

    # Memory monitoring
    memory_info = dm.get_device_memory_info()
    print(f"GPU Memory: {memory_info['allocated_gb']:.2f} GB")

    # Model analysis
    model_info = dm.get_model_size(my_model)
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")

    # Performance testing
    benchmark = dm.benchmark_device(size=(2000, 2000), iterations=50)
    print(f"Average operation time: {benchmark['avg_time_ms']} ms")

    # Inference optimization
    with dm.inference_mode():
        # Optimized settings for inference
        predictions = model(test_data)

    # System information
    dm.print_system_info()  # Comprehensive system report
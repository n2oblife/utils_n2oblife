import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Generator
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders"""
    
    def __init__(self, path: str, batch_size: int = 32, shuffle: bool = True):
        self.path = Path(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_paths = []
        self.labels = []
        self.categories = []
        self._current_index = 0
        
        if not self.path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
            
        self._scan_directory()
        
    def _scan_directory(self):
        """Scan directory structure and collect file paths without loading data"""
        self.categories = [d.name for d in self.path.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
        
        if not self.categories:
            raise ValueError(f"No valid categories found in {self.path}")
            
        logger.info(f"Found categories: {self.categories}")
        
        for category in self.categories:
            category_path = self.path / category
            class_index = self.categories.index(category)
            
            for file_path in category_path.iterdir():
                if self._is_valid_file(file_path):
                    self.file_paths.append(file_path)
                    self.labels.append(class_index)
        
        if self.shuffle:
            self._shuffle_data()
            
        logger.info(f"Found {len(self.file_paths)} valid files")
    
    def _shuffle_data(self):
        """Shuffle file paths and labels together"""
        indices = np.random.permutation(len(self.file_paths))
        self.file_paths = [self.file_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
    
    @abstractmethod
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is valid for this data type"""
        pass
    
    @abstractmethod
    def _load_single_item(self, file_path: Path) -> np.ndarray:
        """Load a single data item from file"""
        pass
    
    def __len__(self) -> int:
        """Return number of batches"""
        return (len(self.file_paths) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Make the dataloader iterable"""
        self._current_index = 0
        if self.shuffle:
            self._shuffle_data()
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch"""
        if self._current_index >= len(self.file_paths):
            raise StopIteration
            
        batch_data, batch_labels = self._get_batch()
        return batch_data, batch_labels
    
    def _get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return a batch of data"""
        start_idx = self._current_index
        end_idx = min(start_idx + self.batch_size, len(self.file_paths))
        
        batch_files = self.file_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        batch_data = []
        valid_labels = []
        
        for file_path, label in zip(batch_files, batch_labels):
            try:
                data = self._load_single_item(file_path)
                if data is not None:
                    batch_data.append(data)
                    valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        self._current_index = end_idx
        
        if not batch_data:
            raise ValueError("No valid data in batch")
            
        return np.array(batch_data), np.array(valid_labels)
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Get a single sample by index"""
        if index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of range")
            
        file_path = self.file_paths[index]
        label = self.labels[index]
        data = self._load_single_item(file_path)
        return data, label
    
    def get_class_names(self) -> List[str]:
        """Return list of class names"""
        return self.categories
    
    def get_dataset_info(self) -> dict:
        """Return dataset information"""
        return {
            'num_samples': len(self.file_paths),
            'num_classes': len(self.categories),
            'class_names': self.categories,
            'batch_size': self.batch_size,
            'num_batches': len(self)
        }


class ImageDataLoader(BaseDataLoader):
    """DataLoader for image classification tasks"""
    
    def __init__(self, path: str, image_size: int = 224, batch_size: int = 32, 
                 shuffle: bool = True, normalize: bool = True, 
                 valid_extensions: List[str] = None):
        self.image_size = image_size
        self.normalize = normalize
        self.valid_extensions = valid_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        super().__init__(path, batch_size, shuffle)
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is a valid image"""
        return file_path.suffix.lower() in self.valid_extensions
    
    def _load_single_item(self, file_path: Path) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Read image
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not read image: {file_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            # Normalize if requested
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None


class SegmentationDataLoader(BaseDataLoader):
    """DataLoader for image segmentation tasks"""
    
    def __init__(self, path: str, image_size: int = 224, batch_size: int = 32,
                 shuffle: bool = True, mask_suffix: str = '_mask'):
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        super().__init__(path, batch_size, shuffle)
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if both image and mask exist"""
        if self.mask_suffix in file_path.stem:
            return False  # Skip mask files in initial scan
        
        mask_path = file_path.parent / f"{file_path.stem}{self.mask_suffix}{file_path.suffix}"
        return (file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] and 
                mask_path.exists())
    
    def _load_single_item(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and corresponding mask"""
        try:
            # Load image
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            
            # Load mask
            mask_path = file_path.parent / f"{file_path.stem}{self.mask_suffix}{file_path.suffix}"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = mask.astype(np.float32) / 255.0
            
            return np.concatenate([image, mask[..., np.newaxis]], axis=-1)
            
        except Exception as e:
            logger.error(f"Error loading segmentation data {file_path}: {e}")
            return None


class TextDataLoader(BaseDataLoader):
    """DataLoader for text classification tasks"""
    
    def __init__(self, path: str, batch_size: int = 32, shuffle: bool = True,
                 max_length: int = 512, encoding: str = 'utf-8'):
        self.max_length = max_length
        self.encoding = encoding
        super().__init__(path, batch_size, shuffle)
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is a valid text file"""
        return file_path.suffix.lower() in ['.txt', '.md', '.json']
    
    def _load_single_item(self, file_path: Path) -> str:
        """Load text from file"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                text = f.read()
            
            # Truncate if too long
            if len(text) > self.max_length:
                text = text[:self.max_length]
                
            return text
            
        except Exception as e:
            logger.error(f"Error loading text {file_path}: {e}")
            return None


class DataLoaderFactory:
    """Factory class to create appropriate dataloader based on task type"""
    
    @staticmethod
    def create_dataloader(task_type: str, path: str, **kwargs) -> BaseDataLoader:
        """Create dataloader based on task type"""
        
        task_type = task_type.lower()
        
        if task_type in ['classification', 'image_classification']:
            return ImageDataLoader(path, **kwargs)
        elif task_type in ['segmentation', 'image_segmentation']:
            return SegmentationDataLoader(path, **kwargs)
        elif task_type in ['text_classification', 'text']:
            return TextDataLoader(path, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


# Example usage and testing
if __name__ == "__main__":
    # Example usage for image classification
    try:
        # Create dataloader
        dataloader = DataLoaderFactory.create_dataloader(
            task_type='classification',
            path='./data',
            image_size=224,
            batch_size=16,
            shuffle=True
        )
        
        # Print dataset info
        print("Dataset Info:", dataloader.get_dataset_info())
        
        # Iterate through batches
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Data shape: {batch_data.shape}, Labels shape: {batch_labels.shape}")
            
            # Process only first few batches for demo
            if batch_idx >= 2:
                break
                
        # Get a single sample
        sample_data, sample_label = dataloader.get_sample(0)
        print(f"Sample shape: {sample_data.shape}, Label: {sample_label}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a data directory with subdirectories containing images")
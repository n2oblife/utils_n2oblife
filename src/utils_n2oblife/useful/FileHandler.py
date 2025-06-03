import os
import pickle
import shutil
import json
import csv
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
import logging


class FileHandler:
    """
    A comprehensive file handling class that provides convenient methods for 
    file operations, data serialization, and file system management.
    
    This class simplifies common file operations and supports multiple data formats
    including pickle, numpy, JSON, CSV, and plain text files.
    """
    
    def __init__(self, base_path: Optional[str] = None, create_dirs: bool = True, 
                 verbose: bool = True, backup: bool = False):
        """
        Initialize the FileHandler instance.
        
        Args:
            base_path (str, optional): Base directory for relative paths (default: current directory)
            create_dirs (bool): Automatically create directories if they don't exist (default: True)
            verbose (bool): Print operation messages (default: True)
            backup (bool): Create backups before overwriting files (default: False)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.create_dirs = create_dirs
        self.verbose = verbose
        self.backup = backup
        self.supported_formats = {
            'pickle': {'ext': '.pkl', 'binary': True},
            'npy': {'ext': '.npy', 'binary': True},
            'json': {'ext': '.json', 'binary': False},
            'csv': {'ext': '.csv', 'binary': False},
            'txt': {'ext': '.txt', 'binary': False}
        }
        
        # Ensure base path exists
        if self.create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, filename: str) -> Path:
        """
        Get the full path for a given filename.
        
        Args:
            filename (str): Filename or relative path
            
        Returns:
            Path: Full path object
        """
        path = Path(filename)
        if not path.is_absolute():
            path = self.base_path / path
        return path
    
    def _ensure_directory(self, filepath: Path) -> None:
        """
        Ensure the directory for a file path exists.
        
        Args:
            filepath (Path): File path to check
        """
        if self.create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def _create_backup(self, filepath: Path) -> Optional[Path]:
        """
        Create a backup of an existing file.
        
        Args:
            filepath (Path): Path to the file to backup
            
        Returns:
            Path or None: Path to backup file if created
        """
        if filepath.exists() and self.backup:
            backup_path = filepath.with_suffix(f'{filepath.suffix}.bak')
            shutil.copy2(filepath, backup_path)
            if self.verbose:
                print(f"Backup created: {backup_path}")
            return backup_path
        return None
    
    def _log_operation(self, operation: str, filepath: Path, success: bool = True) -> None:
        """
        Log file operations if verbose mode is enabled.
        
        Args:
            operation (str): Operation description
            filepath (Path): File path involved
            success (bool): Whether operation was successful
        """
        if self.verbose:
            status = "✓" if success else "✗"
            print(f"{status} {operation}: {filepath}")
    
    def save_data(self, data: Any, filename: str, format: str = 'pickle', 
                  **kwargs) -> bool:
        """
        Save data to a file in the specified format.
        
        Args:
            data: The data to save
            filename (str): The filename to save the data to
            format (str): The format to save the data in (default: 'pickle')
            **kwargs: Additional arguments for specific formats
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If format is not supported
            IOError: If file operations fail
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(self.supported_formats.keys())}")
        
        filepath = self._get_full_path(filename)
        self._ensure_directory(filepath)
        self._create_backup(filepath)
        
        try:
            if format == 'pickle':
                with open(filepath, 'wb') as file:
                    pickle.dump(data, file, protocol=kwargs.get('protocol', pickle.HIGHEST_PROTOCOL))
            
            elif format == 'npy':
                np.save(filepath, data, **kwargs)
            
            elif format == 'json':
                with open(filepath, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=kwargs.get('indent', 2), 
                             ensure_ascii=kwargs.get('ensure_ascii', False), **kwargs)
            
            elif format == 'csv':
                with open(filepath, 'w', newline='', encoding='utf-8') as file:
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict):
                            # List of dictionaries
                            writer = csv.DictWriter(file, fieldnames=data[0].keys(), **kwargs)
                            writer.writeheader()
                            writer.writerows(data)
                        else:
                            # List of lists/tuples
                            writer = csv.writer(file, **kwargs)
                            writer.writerows(data)
                    else:
                        raise ValueError("CSV data must be a list of dictionaries or list of lists/tuples")
            
            elif format == 'txt':
                mode = 'w' if isinstance(data, str) else 'wb'
                encoding = 'utf-8' if isinstance(data, str) else None
                with open(filepath, mode, encoding=encoding) as file:
                    file.write(data)
            
            self._log_operation(f"Saved ({format})", filepath)
            return True
            
        except Exception as e:
            self._log_operation(f"Failed to save ({format})", filepath, success=False)
            raise IOError(f"Failed to save {filename}: {str(e)}")
    
    def load_data(self, filename: str, format: str = 'pickle', **kwargs) -> Any:
        """
        Load data from a file in the specified format.
        
        Args:
            filename (str): The filename to load the data from
            format (str): The format to load the data in (default: 'pickle')
            **kwargs: Additional arguments for specific formats
            
        Returns:
            Any: The loaded data
            
        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
            IOError: If file operations fail
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(self.supported_formats.keys())}")
        
        filepath = self._get_full_path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            if format == 'pickle':
                with open(filepath, 'rb') as file:
                    data = pickle.load(file)
            
            elif format == 'npy':
                data = np.load(filepath, **kwargs)
            
            elif format == 'json':
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file, **kwargs)
            
            elif format == 'csv':
                data = []
                with open(filepath, 'r', encoding='utf-8') as file:
                    if kwargs.get('dict_reader', True):
                        reader = csv.DictReader(file, **{k: v for k, v in kwargs.items() if k != 'dict_reader'})
                        data = list(reader)
                    else:
                        reader = csv.reader(file, **kwargs)
                        data = list(reader)
            
            elif format == 'txt':
                mode = 'r' if kwargs.get('text_mode', True) else 'rb'
                encoding = 'utf-8' if mode == 'r' else None
                with open(filepath, mode, encoding=encoding) as file:
                    data = file.read()
            
            self._log_operation(f"Loaded ({format})", filepath)
            return data
            
        except Exception as e:
            self._log_operation(f"Failed to load ({format})", filepath, success=False)
            raise IOError(f"Failed to load {filename}: {str(e)}")
    
    def copy_file(self, src: str, dst: str, preserve_metadata: bool = True) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            src (str): Source file path
            dst (str): Destination file path
            preserve_metadata (bool): Preserve file metadata (default: True)
            
        Returns:
            bool: True if successful, False otherwise
        """
        src_path = self._get_full_path(src)
        dst_path = self._get_full_path(dst)
        
        self._ensure_directory(dst_path)
        self._create_backup(dst_path)
        
        try:
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
            
            self._log_operation("Copied", f"{src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            self._log_operation("Failed to copy", f"{src_path} -> {dst_path}", success=False)
            if self.verbose:
                print(f"Error: {str(e)}")
            return False
    
    def move_file(self, src: str, dst: str) -> bool:
        """
        Move a file from source to destination.
        
        Args:
            src (str): Source file path
            dst (str): Destination file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        src_path = self._get_full_path(src)
        dst_path = self._get_full_path(dst)
        
        self._ensure_directory(dst_path)
        
        try:
            shutil.move(str(src_path), str(dst_path))
            self._log_operation("Moved", f"{src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            self._log_operation("Failed to move", f"{src_path} -> {dst_path}", success=False)
            if self.verbose:
                print(f"Error: {str(e)}")
            return False
    
    def delete_file(self, filename: str, safe: bool = True) -> bool:
        """
        Delete a file.
        
        Args:
            filename (str): File to delete
            safe (bool): Move to trash/backup instead of permanent deletion (default: True)
            
        Returns:
            bool: True if successful, False otherwise
        """
        filepath = self._get_full_path(filename)
        
        if not filepath.exists():
            if self.verbose:
                print(f"File not found: {filepath}")
            return False
        
        try:
            if safe and self.backup:
                # Create backup before deletion
                backup_path = filepath.with_suffix(f'{filepath.suffix}.deleted')
                shutil.move(str(filepath), str(backup_path))
                self._log_operation("Moved to backup", f"{filepath} -> {backup_path}")
            else:
                filepath.unlink()
                self._log_operation("Deleted", filepath)
            
            return True
            
        except Exception as e:
            self._log_operation("Failed to delete", filepath, success=False)
            if self.verbose:
                print(f"Error: {str(e)}")
            return False
    
    def check_files_exist(self, filenames: Union[str, List[str]]) -> Union[bool, Dict[str, bool]]:
        """
        Check if specified files exist.
        
        Args:
            filenames (Union[str, List[str]]): Single filename or list of filenames to check
            
        Returns:
            Union[bool, Dict[str, bool]]: Boolean for single file, dict for multiple files
        """
        if isinstance(filenames, str):
            filepath = self._get_full_path(filenames)
            return filepath.exists()
        
        files_exist = {}
        for filename in filenames:
            filepath = self._get_full_path(filename)
            files_exist[filename] = filepath.exists()
        
        return files_exist
    
    def check_folder_exists(self, folder_path: str) -> bool:
        """
        Check if the specified folder exists.
        
        Args:
            folder_path (str): The path to the folder
            
        Returns:
            bool: True if the folder exists, False otherwise
        """
        folderpath = self._get_full_path(folder_path)
        return folderpath.is_dir()
    
    def create_folder(self, folder_path: str, exist_ok: bool = True) -> bool:
        """
        Create a folder and any necessary parent directories.
        
        Args:
            folder_path (str): Path to the folder to create
            exist_ok (bool): Don't raise error if folder already exists (default: True)
            
        Returns:
            bool: True if successful, False otherwise
        """
        folderpath = self._get_full_path(folder_path)
        
        try:
            folderpath.mkdir(parents=True, exist_ok=exist_ok)
            self._log_operation("Created folder", folderpath)
            return True
            
        except Exception as e:
            self._log_operation("Failed to create folder", folderpath, success=False)
            if self.verbose:
                print(f"Error: {str(e)}")
            return False
    
    def list_files(self, pattern: str = "*", recursive: bool = False, 
                   files_only: bool = True) -> List[Path]:
        """
        List files in the base directory matching a pattern.
        
        Args:
            pattern (str): Glob pattern to match (default: "*")
            recursive (bool): Search recursively (default: False)
            files_only (bool): Return only files, not directories (default: True)
            
        Returns:
            List[Path]: List of matching file paths
        """
        if recursive:
            paths = self.base_path.rglob(pattern)
        else:
            paths = self.base_path.glob(pattern)
        
        if files_only:
            return [p for p in paths if p.is_file()]
        else:
            return list(paths)
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get detailed information about a file.
        
        Args:
            filename (str): File to analyze
            
        Returns:
            Dict[str, Any]: File information including size, dates, etc.
        """
        filepath = self._get_full_path(filename)
        
        if not filepath.exists():
            return {'exists': False}
        
        stat = filepath.stat()
        
        return {
            'exists': True,
            'path': str(filepath),
            'size': stat.st_size,
            'size_human': self._human_readable_size(stat.st_size),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'accessed': stat.st_atime,
            'is_file': filepath.is_file(),
            'is_dir': filepath.is_dir(),
            'suffix': filepath.suffix,
            'name': filepath.name,
            'stem': filepath.stem
        }
    
    def _human_readable_size(self, size: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    @contextmanager
    def batch_operations(self, verbose: bool = False):
        """
        Context manager for batch operations with reduced verbosity.
        
        Args:
            verbose (bool): Override verbosity for batch operations
        """
        original_verbose = self.verbose
        self.verbose = verbose
        try:
            yield self
        finally:
            self.verbose = original_verbose
    
    def search_in_files(self, pattern: str, file_pattern: str = "*.txt", 
                       case_sensitive: bool = False) -> Dict[str, List[int]]:
        """
        Search for a pattern in text files.
        
        Args:
            pattern (str): Text pattern to search for
            file_pattern (str): File pattern to search in (default: "*.txt")
            case_sensitive (bool): Case sensitive search (default: False)
            
        Returns:
            Dict[str, List[int]]: Dictionary mapping filenames to line numbers containing pattern
        """
        import re
        
        results = {}
        files = self.list_files(file_pattern, recursive=True)
        
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    matches = []
                    for line_num, line in enumerate(file, 1):
                        if regex.search(line):
                            matches.append(line_num)
                    
                    if matches:
                        results[str(filepath.relative_to(self.base_path))] = matches
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error searching in {filepath}: {e}")
        
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Could add cleanup operations here if needed
        pass
    
    def __repr__(self) -> str:
        """String representation of the FileHandler instance."""
        return f"FileHandler(base_path='{self.base_path}', verbose={self.verbose})"
    
if __name__ == "__main__":
    # Basic usage
    fh = FileHandler("/my/project/data", verbose=True, backup=True)

    # Save different formats
    fh.save_data(my_dict, "config.json", format="json")
    fh.save_data(numpy_array, "data.npy", format="npy")
    fh.save_data(csv_data, "results.csv", format="csv")

    # Load data
    config = fh.load_data("config.json", format="json")
    data = fh.load_data("data.npy", format="npy")

    # File operations
    fh.copy_file("source.txt", "backup/source.txt")
    fh.move_file("temp.txt", "archive/temp.txt")

    # Batch operations with reduced verbosity
    with fh.batch_operations(verbose=False) as batch_fh:
        for i in range(100):
            batch_fh.save_data(f"data_{i}", f"file_{i}.pkl")

    # Context manager usage
    with FileHandler("/project/data") as fh:
        data = fh.load_data("input.pkl")
        processed = process_data(data)
        fh.save_data(processed, "output.pkl")

    # Search operations
    files = fh.list_files("*.py", recursive=True)
    matches = fh.search_in_files("TODO", "*.py")

    # File information
    info = fh.get_file_info("large_dataset.npy")
    print(f"File size: {info['size_human']}")
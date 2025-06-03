"""
Optional Dependencies Management System

This module provides utilities for handling optional dependencies in Python packages,
allowing users to install only the requirements they need for specific functionality.
"""

import sys
import importlib
import subprocess
from typing import Dict, List, Optional, Union, Any, Callable
from functools import wraps
import warnings


class DependencyError(ImportError):
    """Custom exception for missing optional dependencies."""
    pass


class OptionalDependency:
    """Manages optional dependencies with lazy loading and helpful error messages."""
    
    def __init__(self, 
                 package_name: str,
                 import_name: str = None,
                 pip_name: str = None,
                 conda_name: str = None,
                 min_version: str = None,
                 install_command: str = None):
        """
        Initialize an optional dependency.
        
        Args:
            package_name: Name used for importing (e.g., 'sklearn')
            import_name: Alternative import name if different from package_name
            pip_name: Name for pip installation (e.g., 'scikit-learn')
            conda_name: Name for conda installation
            min_version: Minimum required version
            install_command: Custom installation command
        """
        self.package_name = package_name
        self.import_name = import_name or package_name
        self.pip_name = pip_name or package_name
        self.conda_name = conda_name or self.pip_name
        self.min_version = min_version
        self.install_command = install_command
        self._module = None
        self._checked = False
        self._available = False
    
    def is_available(self) -> bool:
        """Check if the dependency is available."""
        if not self._checked:
            try:
                self._module = importlib.import_module(self.import_name)
                self._check_version()
                self._available = True
            except ImportError:
                self._available = False
            finally:
                self._checked = True
        return self._available
    
    def _check_version(self):
        """Check if the installed version meets minimum requirements."""
        if self.min_version and self._module:
            try:
                import packaging.version
                installed_version = getattr(self._module, '__version__', '0.0.0')
                if packaging.version.parse(installed_version) < packaging.version.parse(self.min_version):
                    raise DependencyError(
                        f"{self.package_name} version {installed_version} is installed, "
                        f"but version {self.min_version} or higher is required"
                    )
            except ImportError:
                # packaging not available, skip version check
                pass
    
    def get_module(self):
        """Get the imported module or raise helpful error."""
        if self.is_available():
            return self._module
        else:
            self._raise_dependency_error()
    
    def _raise_dependency_error(self):
        """Raise a helpful error message for missing dependency."""
        error_msg = f"Missing optional dependency '{self.package_name}'"
        
        if self.install_command:
            install_msg = f"Install with: {self.install_command}"
        else:
            install_options = []
            install_options.append(f"pip install {self.pip_name}")
            if self.conda_name != self.pip_name:
                install_options.append(f"conda install {self.conda_name}")
            install_msg = "Install with: " + " or ".join(install_options)
        
        if self.min_version:
            install_msg += f" (>={self.min_version})"
        
        raise DependencyError(f"{error_msg}. {install_msg}")
    
    def __getattr__(self, name):
        """Allow direct access to module attributes."""
        module = self.get_module()
        return getattr(module, name)


class DependencyManager:
    """Centralized manager for all optional dependencies."""
    
    def __init__(self):
        self.dependencies: Dict[str, OptionalDependency] = {}
        self.dependency_groups: Dict[str, List[str]] = {}
    
    def register(self, name: str, dependency: OptionalDependency):
        """Register an optional dependency."""
        self.dependencies[name] = dependency
    
    def register_group(self, group_name: str, dependency_names: List[str]):
        """Register a group of dependencies for a feature."""
        self.dependency_groups[group_name] = dependency_names
    
    def get(self, name: str) -> OptionalDependency:
        """Get a registered dependency."""
        if name not in self.dependencies:
            raise KeyError(f"Dependency '{name}' not registered")
        return self.dependencies[name]
    
    def check_group(self, group_name: str) -> Dict[str, bool]:
        """Check availability of all dependencies in a group."""
        if group_name not in self.dependency_groups:
            raise KeyError(f"Dependency group '{group_name}' not registered")
        
        results = {}
        for dep_name in self.dependency_groups[group_name]:
            dep = self.get(dep_name)
            results[dep_name] = dep.is_available()
        return results
    
    def require_group(self, group_name: str):
        """Ensure all dependencies in a group are available."""
        results = self.check_group(group_name)
        missing = [name for name, available in results.items() if not available]
        
        if missing:
            missing_deps = [self.get(name) for name in missing]
            error_msg = f"Missing dependencies for '{group_name}': {', '.join(missing)}\n"
            for dep in missing_deps:
                try:
                    dep._raise_dependency_error()
                except DependencyError as e:
                    error_msg += f"  - {str(e)}\n"
            raise DependencyError(error_msg.strip())
    
    def install_group(self, group_name: str, method: str = 'pip'):
        """Attempt to install missing dependencies in a group."""
        results = self.check_group(group_name)
        missing = [name for name, available in results.items() if not available]
        
        if not missing:
            print(f"All dependencies for '{group_name}' are already installed")
            return
        
        for dep_name in missing:
            dep = self.get(dep_name)
            try:
                if method == 'pip':
                    cmd = f"pip install {dep.pip_name}"
                    if dep.min_version:
                        cmd += f">={dep.min_version}"
                elif method == 'conda':
                    cmd = f"conda install {dep.conda_name}"
                else:
                    cmd = dep.install_command
                
                if cmd:
                    print(f"Installing {dep_name}: {cmd}")
                    subprocess.check_call(cmd.split())
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {dep_name}: {e}")


# Global dependency manager instance
deps = DependencyManager()

# Register common dependencies
deps.register('numpy', OptionalDependency('numpy'))
deps.register('pandas', OptionalDependency('pandas'))
deps.register('sklearn', OptionalDependency(
    package_name='sklearn',
    pip_name='scikit-learn',
    conda_name='scikit-learn'
))
deps.register('torch', OptionalDependency(
    package_name='torch',
    install_command='pip install torch --index-url https://download.pytorch.org/whl/cpu'
))
deps.register('tensorflow', OptionalDependency('tensorflow', min_version='2.0.0'))
deps.register('matplotlib', OptionalDependency('matplotlib'))
deps.register('seaborn', OptionalDependency('seaborn'))
deps.register('plotly', OptionalDependency('plotly'))
deps.register('opencv', OptionalDependency(
    package_name='cv2',
    pip_name='opencv-python',
    conda_name='opencv'
))

# Register dependency groups
deps.register_group('ml_basic', ['numpy', 'pandas', 'sklearn'])
deps.register_group('deep_learning', ['numpy', 'torch'])
deps.register_group('visualization', ['matplotlib', 'seaborn', 'plotly'])
deps.register_group('computer_vision', ['numpy', 'opencv', 'matplotlib'])


def requires_dependencies(*dep_names, group: str = None):
    """
    Decorator to ensure required dependencies are available.
    
    Args:
        *dep_names: Names of individual dependencies required
        group: Name of dependency group required
    """
    def decorator(func_or_class):
        if isinstance(func_or_class, type):
            # Decorating a class
            original_init = func_or_class.__init__
            
            @wraps(original_init)
            def new_init(self, *args, **kwargs):
                _check_dependencies(dep_names, group)
                original_init(self, *args, **kwargs)
            
            func_or_class.__init__ = new_init
            return func_or_class
        else:
            # Decorating a function
            @wraps(func_or_class)
            def wrapper(*args, **kwargs):
                _check_dependencies(dep_names, group)
                return func_or_class(*args, **kwargs)
            return wrapper
    
    return decorator


def _check_dependencies(dep_names, group):
    """Internal function to check dependencies."""
    if group:
        deps.require_group(group)
    
    for dep_name in dep_names:
        dep = deps.get(dep_name)
        dep.get_module()  # This will raise error if not available


def lazy_import(module_name: str, dep_name: str = None):
    """
    Lazy import that only imports when first accessed.
    
    Args:
        module_name: Name of module to import
        dep_name: Name of registered dependency (if different from module_name)
    """
    dep_name = dep_name or module_name
    return LazyImport(module_name, dep_name)


class LazyImport:
    """Lazy import wrapper that delays import until first access."""
    
    def __init__(self, module_name: str, dep_name: str):
        self.module_name = module_name
        self.dep_name = dep_name
        self._module = None
    
    def __getattr__(self, name):
        if self._module is None:
            dep = deps.get(self.dep_name)
            self._module = dep.get_module()
        return getattr(self._module, name)


# Example usage classes demonstrating the system
@requires_dependencies(group='ml_basic')
class MLProcessor:
    """Example class that requires basic ML dependencies."""
    
    def __init__(self):
        # These imports only happen if dependencies are available
        self.np = deps.get('numpy').get_module()
        self.pd = deps.get('pandas').get_module()
        self.sklearn = deps.get('sklearn').get_module()
    
    def process_data(self, data):
        """Process data using ML libraries."""
        df = self.pd.DataFrame(data)
        return self.np.array(df.values)


@requires_dependencies('torch', 'numpy')
class DeepLearningModel:
    """Example class that requires PyTorch."""
    
    def __init__(self):
        self.torch = deps.get('torch').get_module()
        self.np = deps.get('numpy').get_module()
    
    def create_model(self):
        """Create a simple neural network."""
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(10, 5),
            self.torch.nn.ReLU(),
            self.torch.nn.Linear(5, 1)
        )


class VisualizationHelper:
    """Example class with optional visualization capabilities."""
    
    def __init__(self):
        # Use lazy imports for optional features
        self.plt = lazy_import('matplotlib.pyplot', 'matplotlib')
        self.sns = lazy_import('seaborn', 'seaborn')
    
    def plot_data(self, data):
        """Plot data if matplotlib is available."""
        try:
            self.plt.figure(figsize=(10, 6))
            self.plt.plot(data)
            self.plt.show()
        except DependencyError as e:
            print(f"Plotting not available: {e}")
    
    def plot_heatmap(self, data):
        """Create heatmap if seaborn is available."""
        try:
            self.sns.heatmap(data)
            self.plt.show()
        except DependencyError as e:
            print(f"Heatmap not available: {e}")


# Utility functions for checking dependencies
def check_dependencies():
    """Check all registered dependencies and print status."""
    print("Dependency Status:")
    print("-" * 50)
    
    for name, dep in deps.dependencies.items():
        status = "✓ Available" if dep.is_available() else "✗ Missing"
        print(f"{name:15} {status}")
    
    print("\nDependency Groups:")
    print("-" * 50)
    
    for group_name, dep_names in deps.dependency_groups.items():
        results = deps.check_group(group_name)
        available_count = sum(results.values())
        total_count = len(results)
        status = f"{available_count}/{total_count} available"
        print(f"{group_name:15} {status}")


def install_missing_for_group(group_name: str):
    """Install missing dependencies for a specific group."""
    try:
        deps.install_group(group_name)
        print(f"Installation complete for group '{group_name}'")
    except Exception as e:
        print(f"Installation failed: {e}")


if __name__ == "__main__":
    # Example usage
    print("=== Dependency Management System Demo ===\n")
    
    # Check current dependency status
    check_dependencies()
    
    # Try to use classes with different dependency requirements
    print("\n=== Testing Classes ===")
    
    try:
        print("Creating MLProcessor...")
        ml_proc = MLProcessor()
        print("✓ MLProcessor created successfully")
    except DependencyError as e:
        print(f"✗ MLProcessor failed: {e}")
    
    try:
        print("Creating DeepLearningModel...")
        dl_model = DeepLearningModel()
        print("✓ DeepLearningModel created successfully")
    except DependencyError as e:
        print(f"✗ DeepLearningModel failed: {e}")
    
    # Test visualization with graceful degradation
    print("\nTesting VisualizationHelper...")
    viz = VisualizationHelper()
    viz.plot_data([1, 2, 3, 4, 5])  # Will warn if matplotlib not available
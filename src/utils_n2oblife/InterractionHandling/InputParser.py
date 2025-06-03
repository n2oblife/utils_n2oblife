import os
import sys
import warnings
import logging
import argparse
from copy import deepcopy
from typing import Union, List, Any, Optional, Dict


def set_logging_info(mode: str = 'default') -> None:
    """Configure logging settings."""
    if mode == 'default':
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s'
        )
        warnings.simplefilter("ignore")
    else:
        raise ValueError(f"Unsupported logging mode: {mode}")


class InputParser:
    """A class to manage command-line argument parsing options."""
    
    def __init__(self, 
                 long: str, 
                 short: Optional[str] = None, 
                 action: Optional[str] = None,
                 choices: Optional[List[Any]] = None,
                 const: Optional[Any] = None,
                 default: Optional[Any] = None, 
                 dest: Optional[str] = None,
                 help: str = '', 
                 metavar: str = '',
                 nargs: Optional[Union[int, str]] = None,
                 required: bool = False,
                 type: Optional[Union[type, Any]] = None
                 ) -> None:
        """Initialize a ParserOptions instance.

        Args:
            long: The long name of the argument (e.g., 'verbose' for --verbose)
            short: Short name of the argument (e.g., 'v' for -v)
            action: Action to take when argument is encountered
            choices: List of valid choices for the argument
            const: Constant value for store_const and append_const actions
            default: Default value if argument is not provided
            dest: Name of attribute to store parsed value
            help: Help text for the argument
            metavar: Name for argument in usage messages
            nargs: Number of command-line arguments to consume
            required: Whether the argument is required
            type: Type to convert argument to
        """
        # Validate required parameters
        if not long or not isinstance(long, str):
            raise ValueError("'long' parameter must be a non-empty string")
        
        if short is not None and (not isinstance(short, str) or len(short) == 0):
            raise ValueError("'short' parameter must be a non-empty string or None")
        
        self.long = long
        self.short = short
        self.action = action
        self.choices = choices
        self.const = const
        self.default = default
        self.dest = dest or long  # Use long name as dest if not provided
        self.help = help
        self.metavar = metavar
        self.nargs = nargs
        self.required = required
        self.type = type
    
    def __str__(self) -> str:
        """Return string representation of ParserOptions."""
        return (f"ParserOptions(long='{self.long}', short='{self.short}', "
                f"action='{self.action}', default={self.default}, "
                f"required={self.required})")
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        attrs = []
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, str):
                    attrs.append(f"{key}='{value}'")
                else:
                    attrs.append(f"{key}={value}")
        return f"ParserOptions({', '.join(attrs)})"
    
    def __len__(self) -> int:
        """Return number of non-None attributes."""
        return sum(1 for value in self.__dict__.values() if value is not None)
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on all attributes."""
        if not isinstance(other, InputParser):
            return False
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        """Return hash based on immutable attributes."""
        # Only hash immutable attributes to avoid issues with mutable defaults
        hashable_attrs = (
            self.long, self.short, self.action, self.dest, 
            self.help, self.metavar, self.nargs, self.required, self.type
        )
        return hash(hashable_attrs)
    
    def __getitem__(self, key: str) -> Any:
        """Get attribute value by key."""
        if key not in self.__dict__:
            raise KeyError(f"'{key}' not found in ParserOptions")
        return self.__dict__[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set attribute value by key."""
        if key not in self.__dict__:
            raise KeyError(f"'{key}' is not a valid ParserOptions attribute")
        self.__dict__[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete attribute (set to None)."""
        if key not in self.__dict__:
            raise KeyError(f"'{key}' not found in ParserOptions")
        if key in ('long',):  # Protect required attributes
            raise ValueError(f"Cannot delete required attribute '{key}'")
        self.__dict__[key] = None

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not None."""
        return key in self.__dict__ and self.__dict__[key] is not None
    
    def __iter__(self):
        """Iterate over attribute names."""
        return iter(self.__dict__)
    
    def __copy__(self):
        """Create a shallow copy."""
        return InputParser(
            long=self.long,
            short=self.short,
            action=self.action,
            choices=self.choices,
            const=self.const,
            default=self.default,
            dest=self.dest,
            help=self.help,
            metavar=self.metavar,
            nargs=self.nargs,
            required=self.required,
            type=self.type
        )
        
    def __deepcopy__(self, memo):
        """Create a deep copy."""
        return InputParser(
            long=self.long,
            short=self.short,
            action=self.action,
            choices=deepcopy(self.choices, memo) if self.choices else None,
            const=deepcopy(self.const, memo),
            default=deepcopy(self.default, memo),
            dest=self.dest,
            help=self.help,
            metavar=self.metavar,
            nargs=self.nargs,
            required=self.required,
            type=self.type
        )
    
    def to_argparse_kwargs(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for argparse.add_argument()."""
        kwargs = {}
        
        # Map attributes to argparse parameters
        arg_mapping = {
            'action': self.action,
            'choices': self.choices,
            'const': self.const,
            'default': self.default,
            'dest': self.dest,
            'help': self.help,
            'metavar': self.metavar,
            'nargs': self.nargs,
            'required': self.required,
            'type': self.type
        }
        
        # Only include non-None values
        for key, value in arg_mapping.items():
            if value is not None:
                kwargs[key] = value
        
        # Add choices to help text if present
        if self.choices and self.help:
            kwargs['help'] = f"{self.help} (choices: {self.choices})"
        elif self.choices:
            kwargs['help'] = f"Choices: {self.choices}"
        
        # Ensure help text ends with period
        if 'help' in kwargs and kwargs['help'] and not kwargs['help'].endswith('.'):
            kwargs['help'] += '.'
        
        return kwargs
    
    def get_argument_names(self) -> List[str]:
        """Get the argument names for argparse.add_argument()."""
        names = [f'--{self.long}']
        if self.short:
            names.insert(0, f'-{self.short}')
        return names


def add_argument_to_parser(parser: argparse.ArgumentParser, option: InputParser) -> None:
    """Add a ParserOptions instance to an argparse parser."""
    if not isinstance(option, InputParser):
        raise TypeError("option must be a ParserOptions instance")
    
    names = option.get_argument_names()
    kwargs = option.to_argparse_kwargs()
    
    parser.add_argument(*names, **kwargs)


def parse_input(parser_config: Union[InputParser, List[InputParser]], 
                prog_name: str = "",
                description: str = "",
                epilog: str = "",
                formatter_class = argparse.ArgumentDefaultsHelpFormatter
                ) -> Dict[str, Any]:
    """Parse command-line arguments using ParserOptions configuration.

    Args:
        parser_config: Single ParserOptions or list of ParserOptions
        prog_name: Program name for help text
        description: Program description for help text
        epilog: Text to display after argument descriptions
        formatter_class: Formatter class for help text

    Returns:
        Dictionary of parsed arguments

    Raises:
        ValueError: If parser_config is invalid
        TypeError: If parser_config has wrong type
    """
    if not parser_config:
        raise ValueError("parser_config cannot be None or empty")
    
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=description,
        epilog=epilog,
        formatter_class=formatter_class
    )
    
    # Handle single ParserOptions or list
    if isinstance(parser_config, InputParser):
        add_argument_to_parser(parser, parser_config)
    elif isinstance(parser_config, list):
        if not parser_config:
            raise ValueError("parser_config list cannot be empty")
        for option in parser_config:
            if not isinstance(option, InputParser):
                raise TypeError("All items in parser_config list must be ParserOptions instances")
            add_argument_to_parser(parser, option)
    else:
        raise TypeError("parser_config must be ParserOptions or List[ParserOptions]")
    
    return vars(parser.parse_args())


def build_example_args() -> Dict[str, Any]:
    """
    Example function showing how to build and parse command-line arguments.
    
    Returns:
        Dictionary of parsed arguments
    """
    parser_options = [
        InputParser(
            long='folder_path',
            short='f',
            type=str,
            required=True,
            help='Path to the input folder'
        ),
        InputParser(
            long='width',
            short='w',
            type=int,
            default=800,
            help='Width of the output'
        ),
        InputParser(
            long='height',
            short='h',
            type=int,
            default=600,
            help='Height of the output'
        ),
        InputParser(
            long='depth',
            short='d',
            type=int,
            default=10,
            help='Depth parameter'
        ),
        InputParser(
            long='num_frames',
            short='n',
            type=int,
            default=30,
            help='Number of frames'
        ),
        InputParser(
            long='framerate',
            short='r',
            type=float,
            default=24.0,
            help='Frame rate in fps'
        ),
        InputParser(
            long='show_video',
            short='s',
            action='store_true',
            help='Show video output'
        )
    ]
    
    args = parse_input(
        parser_config=parser_options,
        prog_name='Example Program',
        description='An example program demonstrating the parser',
        epilog='For more information, visit our documentation.'
    )
    
    print("--- Input parsed ---")
    return args



# Example usage
if __name__ == "__main__":
    # Example of how to use the refined parser
    try:
        # Set up logging
        set_logging_info()
        
        # Create parser options
        options = [
            ParserOptions(
                long='input',
                short='i',
                type=str,
                required=True,
                help='Input file path'
            ),
            ParserOptions(
                long='output',
                short='o',
                type=str,
                default='output.txt',
                help='Output file path'
            ),
            ParserOptions(
                long='verbose',
                short='v',
                action='store_true',
                help='Enable verbose output'
            )
        ]
        
        # Parse arguments
        args = parse_input(
            parser_config=options,
            prog_name='Refined Parser Example',
            description='Demonstration of the refined parser functionality'
        )
        
        print("Parsed arguments:", args)
        print(f"Python version: {get_python_version()}")
        print(f"In virtual environment: {in_venv()}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
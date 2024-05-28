import os, sys, warnings, logging
import argparse
from copy import deepcopy
from typing import Union, List, Any

# def restart_line():
#     sys.stdout.write('\r')
#     sys.stdout.flush()

def set_logging_info(mode='default')->None:
    if mode=='default':
        logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s')
        warnings.simplefilter("ignore")

class ParserOptions:
    def __init__(self, 
                 long: str, 
                 short: str = None, 
                 action: str = None,
                 choices: List[Any] = None,
                 const: Any = None,
                 default: Any = None, 
                 dest: str = None,
                 help: str = '', 
                 metavar: str = '',
                 nargs: Union[int, str] = None,
                 required: bool = False,
                 type: Union[type, Any] = None
                 ) -> None:
        """A custom class to help manage the parsing of input for python script and use them as arguments.

        Args:
            long (str): The actual name of the arg.
            short (str, optional): A letter or two for arg. Defaults to None.
            action (str, optional): The basic type of action to be taken when this argument is encountered at the command line.
            choices (list[any], optional): A list of choices the args that can be used. Default to None.
            const (any, optional): A constant value required by some action and nargs selections.
            default (Any, optional): A default arg to give. Defaults to None.
            dest (str, optional): The name of the attribute to be added to the object returned by parse_args(). Defaults to None.
            help (str, optional): Help to better understand the use of the arg. Defaults to ''.
            metavar (str, optional): A name for the argument in usage messages. Defaults and prefered to ''.
            nargs (int or str, optional): The number of command-line arguments that should be consumed. Defaults to None.
            required (bool, optional): Indicate whether an argument is required or optional. Defaults to False.
            type (type or function, optional): Automatically convert an argument to the given type. Defaults to None.
        """
        self.long = long
        self.short = short
        self.action = action
        self.choices = choices
        self.const = const
        self.default = default
        self.dest = dest
        self.help = help
        self.metavar = metavar
        self.nargs = nargs
        self.required = required
        self.type = type
    
    def __str__(self) -> str:
        return f"ParserOptions(long={self.long}, short={self.short}, action={self.action}, choices={self.choices}, const={self.const}, default={self.default}, dest={self.dest}, help={self.help}, metavar={self.metavar}, nargs={self.nargs}, required={self.required}, type={self.type}"
    
    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __eq__(self, o: object) -> bool:
        return self.len() == o.len()
    
    def __ne__(self, o: object) -> bool:
        return self.len() != o.len()
    
    def __lt__(self, o: object) -> bool:
        return self.len() < o.len()
    
    def __le__(self, o: object) -> bool:
        return self.len() <= o.len()
    
    def __gt__(self, o: object) -> bool:
        return self.len() > o.len()
    
    def __ge__(self, o: object) -> bool:
        return self.len() >= o.len()
    
    def __hash__(self) -> int:
        return hash(self.__dict__)
    
    def __getitem__(self, key: str) -> str:
        return self.__dict__[key]
    
    def __setitem__(self, key: str, value: str) -> None:
        self.__dict__[key] = value

    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __copy__(self):
        return ParserOptions(
            self.long, 
            self.short, 
            self.action, 
            self.choices, 
            self.const, 
            self.default,
            self.dest, 
            self.help, 
            self.metavar, 
            self.nargs, 
            self.required, 
            self.type
        )
        
    def __deepcopy__(self, memo):
        """Create a deep copy of this ParserOptions instance."""
        return ParserOptions(
            long=self.long,
            short=self.short,
            action=self.action,
            choices=deepcopy(self.choices, memo),
            const=deepcopy(self.const, memo),
            default=deepcopy(self.default, memo),
            dest=self.dest,
            help=self.help,
            metavar=self.metavar,
            nargs=self.nargs,
            required=self.required,
            type=self.type
        )


def parse_input(parser_config: Union[ParserOptions, List[ParserOptions]] = None, 
                prog_name: str = "",
                descr: str = "",
                epilog: str = "",
                mode: str = 'default'
                ) -> dict[str, Any]:
    """An easier way to parse the inputs of python script.

    Args:
        parser_config (ParserOptions or list[ParserOptions]): Configuration for the parser. Defaults to None.
        prog_name (str, optional): Program name. Defaults to "".
        descr (str, optional): Program description. Defaults to "".
        epilog (str, optional): Epilog description. Defaults to "".
        mode (str, optional): Help formatter mode. Defaults to 'default'.

    Returns:
        dict: A dictionary with the input parsed.
    """
    if mode == 'default':
        parser = argparse.ArgumentParser(
            prog=prog_name,
            description=descr,
            epilog=epilog,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        raise NotImplementedError("The non-default case has not been implemented yet")
    
    if parser_config:
        if isinstance(parser_config, ParserOptions):
            add_argument_to_parser(parser, parser_config)
        elif isinstance(parser_config, list):
            for option in parser_config:
                add_argument_to_parser(parser, option)
        else:
            raise ValueError("parser_config must be ParserOptions or a list of ParserOptions")
    else:
        raise ValueError("Need to give ParserOptions or a list of ParserOptions")
    
    return vars(parser.parse_args())

def add_argument_to_parser(parser, option: ParserOptions):
    """Helper function to add an argument to the parser."""
    args = {
        'action': option.action,
        'choices': option.choices,
        'const': option.const,
        'default': option.default,
        'dest': option.dest,
        'help': option.help+'.',
        'metavar': option.metavar,
        'nargs': option.nargs,
        'required': option.required,
        'type': option.type
    }

    # add choices in the help text without too much verbose
    if args['choices']:
        args['help'] += f"\n(choices : {args['choices']})"
    
    # Remove None values from args dictionary
    args = {key: value for key, value in args.items() if value is not None}

    if option.short:
        parser.add_argument(f'-{option.short}', f'--{option.long}', **args)
    else:
        parser.add_argument(f'--{option.long}', **args)


def in_venv():
    """Get base/real prefix, or sys.prefix if there is none."""
    base_prfix = (getattr(sys, "base_prefix", None)
                  or getattr(sys, "real_prefix", None )
                  or sys.prefix
                  )
    return sys.prefix != base_prfix


def get_python_version():
    try :
        return sys.version.splite(' ')[0]
    except:
        pass
import os, sys, warnings, logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# def restart_line():
#     sys.stdout.write('\r')
#     sys.stdout.flush()

def set_logging_info(mode='default')->None:
    if mode=='default':
        logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s')
        warnings.simplefilter("ignore")

class ParserOptions:
    def __init__(self, 
                 long:str, 
                 short:str, 
                 convert:type|function=None,
                 default:str = None, 
                 choices:list[str]=None, 
                 help:str=None, 
                 required=False
                 ) -> None:
        """A custom class to help manage the parsing of input for python script and use them as arguments.
        
        Args:
            long (str): A letter or two for arg.
            short (str,optional): The actual name of the arg. Defaults to None.
            convert (type|function, optional): Automatically convert an argument to the given type.
            default (str, optional): A default arg to give. Defaults to None.
            choices (list[any], optional): A list of choices the args that can be used. Default to None.
            help (str, optional): Help to better understand the use of the arg. Defaults to ''.
            required (bool, optional): Indicate whether an argument is required or optional
        """
        self.short = short
        self.long = long
        self.type = convert
        self.default = default
        self.choices = choices
        self.help = help
        self.required = required

def parse_input(
        parser_config:ParserOptions|list[ParserOptions]=None, 
        prog_name = "",
        descr = "",
        epilog = "",
        mode='default'
        )-> dict:
    """An easier way to parse the inputs of python script

    Args:
        parser_config (ParserOptions|list[ParserOptions]): _description_. Defaults to None.
        mode (str, optional): Help formatter mode. Defaults to 'default'.

    Returns:
        _type_: _description_
    """
    # TODO finish to deal with the parser
    if mode == 'default':
        parser = ArgumentParser(
            prog=prog_name,
            description=descr,
            epilog=epilog,
            formatter_class=ArgumentDefaultsHelpFormatter)
    else :
        raise NotImplementedError("The non default case has not been implemented yet")
    if parser_config:
        if type(parser_config) == ParserOptions:
            parser.add_argument('-'+parser_config.short, 
                    '--'+parser_config.long, 
                    default=parser_config.default,
                    help=parser_config.help)
        elif type(parser_config) == list:
            for parse_arg in parser_config:
                parser.add_argument('-'+parse_arg.short, 
                                    '--'+parse_arg.long, 
                                    default=parse_arg.default,
                                    help=parse_arg.help)
        return vars(parser.parse_args())
    else:
        raise ValueError("Need to give ParserOptions or a list of ParserOptions")
    
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
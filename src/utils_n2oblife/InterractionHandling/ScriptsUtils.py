import os, sys, warnings, logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def set_logging_info(mode='default')->None:
    if mode=='default':
        logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s')
        warnings.simplefilter("ignore")

class ParserOptions:
    def __init__(self, long:str, short:str, default:str = None, help:str=None) -> None:
        """A custom class to help manage the parsing of input for python script and use them as arguments.
        
        Args:
            long (str): A letter or two for arg.
            short (str,optional): The actual name of the arg. Defaults to None.
            default (str, optional): A default arg to give. Defaults to None.
            help (str, optional): Help to better understand the use of the arg. Defaults to ''.
        """
        self.short = short
        self.long = long
        self.default = default
        self.help = help

def parse_input(parser_config:ParserOptions|list[ParserOptions]=None, mode='default'):
    """An easier way to parse the inputs of python script

    Args:
        parser_config (ParserOptions|list[ParserOptions]): _description_. Defaults to None.
        mode (str, optional): Help formatter mode. Defaults to 'default'.

    Returns:
        _type_: _description_
    """
    if mode == 'default':
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    if parser_config:
        if type(parser_config) == ParserOptions:
            parser.add_argument('-'+parser_config.short, 
                    '--'+parser_config.long, 
                    default=parser_config.default,
                    help=parser_config.help)
        elif type(parser_config) == list[ParserOptions]:
            for parse_arg in parser_config:
                parser.add_argument('-'+parse_arg.short, 
                                    '--'+parse_arg.long, 
                                    default=parse_arg.default,
                                    help=parse_arg.help)
        return vars(parser.parse_args())
    else:
        raise ValueError("Need to give ParserOptions or a list of ParserOptions")
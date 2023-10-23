from utils_n2oblife.ComputerVision.MaskFiller import MaskFiller
from utils_n2oblife.InterractionHandling.ScriptsUtils import *

def main():
    set_logging_info()
    cfg = [ParserOptions(short='d', long='dataset', help='The path to the dataset'),
           ParserOptions(short='o', long='output', default='.', help='The path to save the filled pictures'),
           ]
    args = parse_input(parser_config=cfg)
    dataset_path = args['dataset']
    save_file = args['output']
    filler = MaskFiller(path=dataset_path,
                        save_file=save_file)
    filler.fill_dataset()
    # filler.save_dataset(save_file)

if __name__=="__main__":
    main()
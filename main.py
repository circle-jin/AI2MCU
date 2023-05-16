import argparse
import json
from src.models.onnx_model import OnnxModel

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=True)
    args = parser.add_argument_group('Options')

    args.add_argument(
        '-c',
        '--config_file',
        dest='config_file',
        type=str,
        default=None,
        help='json file is required as input. The example jsonfile is in the ./json',
    )
    return parser.parse_args()

def print_json_information(config_json):
    """Print information of the input JSON file

    Args:
        config_json (json): Contents of input JSON file
    """
    if config_json is not None:
        print('------------------------------------------')
        print("[Input file information]")

        for key, value in config_json.items():
            print(f'   {key} : {value}')
        print('------------------------------------------')

def main():
    args = parse_args()
    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file))
            print_json_information(config)
            input_model_name = config.get('input_model_name')
            input_model_path = config.get('input_model_path')
            
            if config.get('input_model_type', None) == None:
                """
                if the argument is None
                """
                print("mode == None, usage: main.py [-h]")
                return
            if 'onnx' == config.get('input_model_type'):
                print('[onnx] start')
                onnx_crop = OnnxModel()
                onnx_crop.load_model(input_model_name, input_model_path)
                onnx_crop.crop_end_of_onnx_model()
                print('[onnx] finish')

if __name__ == '__main__':
    main()
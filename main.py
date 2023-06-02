import argparse
import json
from src.convert.onnx2drpai import Onnx2Drpai
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
        print("[Input File Information]")

        for key, value in config_json.items():
            print(f'   {key} : {value}')
        print('------------------------------------------')

def main():
    args = parse_args()
    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file))
            
            input_error = "Please write the contents of the json file like json/exmaple.json"
            print_json_information(config)
            if config.get('use_crop', None) == True:
                crop_config = config.get('crop', None)
                if crop_config == None:
                    print("[Error] crop == None, " + input_error)
                    return
                print('[Crop Start]')
                input_model_name = crop_config.get('input_model_name')
                input_model_path = crop_config.get('input_model_path')
                if 'onnx' == crop_config.get('input_model_type'):
                    onnx_crop = OnnxModel()
                    onnx_crop.load_model(input_model_name, input_model_path)
                    onnx_crop.crop_end_of_onnx_model()
                print('[Crop Finish]')
                    
            convert_config = config.get('convert', None)
                    
            if config.get('use_convert', None) == True: 
                convert_config = config.get('convert', None)
                if convert_config == None:
                    print("[Error] convert == None, " + input_error)
                    return
                print('[Convert Start]')
                target_device = convert_config.get('target_device')
                output_path = convert_config.get('output_path')
                renesas = Onnx2Drpai(output_path)
                renesas.convert(target_device)
                print('[Convert Finish]')

if __name__ == '__main__':
    main()
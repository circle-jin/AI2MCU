"""Backend for converting onnx to drpai

To run this, you will need to have the DRP-AI-Translator installed as well.
"""
import os, subprocess
from src.convert.ai2mcu import Ai2Mcu
from src.convert.drp_ai_translator import DRPAIConverter

class Onnx2Drpai(Ai2Mcu):
    def __init__(self, output_path :str):
        super().__init__(output_path)
        
    def convert(self, device_type :str):
        python_file_path = os.path.dirname(os.path.abspath(__file__))
        for model_path in self.crop_models_path:
            
            convert = DRPAIConverter(model_path)
            convert.convert_onnx_to_drpai()

            # Check the convert result
            model_name = model_path.split('/')[-1]
            try:
                convert_result_folder_path = f'/root/drp-ai_translator_release/output/{model_name}'
                file_list = os.listdir(convert_result_folder_path)
                file_count = len(file_list)
            except FileNotFoundError:
                raise Exception("The converted model cannot be found.")
                
            RED = "\033[91m"
            GREEN = "\033[32m"
            RESET = "\033[0m"
            print(f"{model_name}, ", end="")
            if file_count == 22:
                print(GREEN + 'Succeeded' + RESET)
            else:
                print(RED + 'Failed' + RESET)
            
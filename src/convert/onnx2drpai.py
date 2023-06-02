"""Backend for converting onnx to drpai

To run this, you will need to have the DRP-AI-Translator installed as well.
"""
import subprocess
from src.convert.ai2mcu import Ai2Mcu

class Onnx2Drpai(Ai2Mcu):
    def __init__(self, output_path :str):
        super().__init__(output_path)
        
    def convert(self, device_type :str):
        
        for model_path in self.crop_models_path:
            convert_command = f"python3 run_translator_input_image_yolov5_bbox_aimcu_script_onnxname.py {model_path} {device_type}"
            print(f"Converting model {model_path}")
            convert_stream = subprocess.Popen(convert_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            convert_stdout, convert_stderr = convert_stream.communicate()
            convert_stream.wait(None)
            convert_exit_code = convert_stream.returncode
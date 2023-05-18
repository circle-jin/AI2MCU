"""Backend for deploying deep learning models to mcu boards

To run this, you will need to have converter tool of the MCU board installed as well.
"""
import os
class Ai2Mcu:
    def __init__(self, output_path :str):
        self.output_path = output_path
        self.crop_models_path = []
        path_dir = "./output/"
        file_list = os.listdir(path_dir)


        for file_name in file_list:
            abs_path = os.path.abspath(file_name)
            self.crop_models_path.append(abs_path)
        
    def convert(self):
        return
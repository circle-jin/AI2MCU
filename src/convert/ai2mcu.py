"""Backend for deploying deep learning models to mcu boards

To run this, you will need to have converter tool of the MCU board installed as well.
"""
import os
class Ai2Mcu:
    def __init__(self, output_path :str):
        self.output_path = output_path
        self.crop_models_path = []
        path_dir = "./output/"
        
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
            
        file_list = os.listdir(path_dir)
        abs_path = os.path.abspath(path_dir)

        for file_name in file_list:
            path = f"{abs_path}/{file_name}"
            self.crop_models_path.append(path)
        
        # Sort by first digit in model name in ascending order
        try:
            self.crop_models_path = sorted(self.crop_models_path, 
                                        key=lambda x: int(x.split('/')[-1].split('_')[0]))
        except Exception:
            pass

    def convert(self):
        return
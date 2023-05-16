"""Backend for deploying deep learning models to mcu boards

To run this, you will need to have converter tool of the MCU board installed as well.
"""
class Ai2Mcu:
    def __init__(self, output_path :str):
        self.output_path = output_path
        
    def convert(self):
        return
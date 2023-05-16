"""Backend for converting onnx to drpai

To run this, you will need to have the DRP-AI-Translator installed as well.
"""
from ai2mcu import Ai2Mcu

class Onnx2Drpai(Ai2Mcu):
    def convert(self):
        return
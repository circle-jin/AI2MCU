"""Backend for cutting the onnx model

To run this, you will need to have onnx installed as well.
"""
import onnx
import os

from google.protobuf.json_format import MessageToDict
from onnx.utils import extract_model
from typing import List, Dict


class OnnxModel:
    def __init__(self):
        self.model_name: str = None
        self.model: onnx.onnx_ml_pb2.ModelProto = None
        self.model_path: str = None
        self.input_names: List[str]  = None
        self.output_names: List[str] = None
        self.nodes: Dict = None # Node graph of an ONNX model

    def load_model(self, model_name: str, model_path: str):
        """Get data from an ONNX model

        Args:
            model_path (str): path of ONNX model
        """
        onnx_model = onnx.load(model_path)
        graph = onnx_model.graph
        
        model_input_names = []
        for input in graph.input:
            input = MessageToDict(input)
            model_input_names.append(input['name'])

        model_output_names = []
        for output in graph.output:
            output = MessageToDict(output)
            model_output_names.append(output['name'])
            
        model_nodes = dict()
        for n in graph.node:
            model_nodes[n.name]=n
            
        self.model_name = model_name
        self.model_path = model_path
        self.model = onnx_model
        self.input_names = model_input_names
        self.output_names = model_output_names
        self.nodes = model_nodes
        
    def crop_end_of_onnx_model(self):
        """Cut one node from the end of the onnx model
        """
        crop_model_dir = './output'
        if not os.path.exists(crop_model_dir):
            os.makedirs(crop_model_dir)
            
        nodes_reversed = list(reversed(self.nodes.items()))
        length = len(nodes_reversed)
        save_index = 0
        for node_index in range(0, length):
            node_info = MessageToDict(nodes_reversed[node_index][1])
            node_opType = node_info['opType']
            
            if node_opType == 'Constant':
                continue
        
            new_output_names = [node_info['output'][0]]
            output_path = f"{crop_model_dir}/{save_index}_{self.model_name}_{node_opType}.onnx"
            # Save a model that cuts one node at the end of the onnx model
            extract_model(self.model_path, output_path, self.input_names, new_output_names)
            print(f'   [{save_index}] save_model : {output_path}')
            save_index += 1
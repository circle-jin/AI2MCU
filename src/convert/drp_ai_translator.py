"""Onnx -> DRPAI model conversion script
"""

import sys, os, argparse
import onnxruntime as ort
from typing import Dict, List, Tuple

TRANSLATOR = os.getenv("TRANSLATOR")
if TRANSLATOR == None:
    print("[Error] No environment variable")
    print("        Before running this script,")
    print("        Please set environment variable(TRANSLATOR)")
    print("        to the directory where you installed DRP-AI Translator")
    print("        e.g. $export TRANSLATOR=/home/user/drp-ai_translator_release/")
    sys.exit(-1)
DRP_TOOLS = TRANSLATOR + "DRP-AI_translator"
sys.path.append(DRP_TOOLS)
import python_api as drp_ai  # type:ignore


class DRPAIConverter:
    CONVERSION_RESULT_DIR_PATH = os.path.join(TRANSLATOR, "output")

    def __init__(self, onnx_model_path: str):
        self.drp_input_name = None
        self.drp_input_type = None
        self.drp_input_height = None
        self.drp_input_width = None
        self.drp_prepost = drp_ai.drp_prepost()
        self.model_path = onnx_model_path
        self.onnx_io_info: dict = self._get_input_output_info_from_model(self.model_path)
        self.drp_tran = drp_ai.drp_ai_translator()
        self.drp_tran.set_translator("V2M")
        
    def convert_onnx_to_drpai(self, target_device: str = "v2l"):
        model_dir_name = os.path.basename(self.model_path)
        drp_input_shape = self.onnx_io_info['input']['shapes'][0]
        self._set_drpai_input(drp_input_shape)
        self._set_drpai_preprocess()
        self._set_drpai_postprocess()
        self._save_drpai_prepost_yaml()
        self._save_drpai_address_yaml(target_device)
        self._run_drpai_translate(model_dir_name=model_dir_name)
        

    def _get_input_output_info_from_model(self, onnx_model_path: str) -> Dict[str, Dict[str, List]]:
        """Get input, outputs information for the model"""
        onnx_model = ort.InferenceSession(path_or_bytes=onnx_model_path)

        input_names = [input.name for input in onnx_model.get_inputs()]
        input_shapes = [input.shape for input in onnx_model.get_inputs()]
        output_names = [output.name for output in onnx_model.get_outputs()]
        output_shapes = [output.shape for output in onnx_model.get_outputs()]

        input_info = {"names": input_names, "shapes": input_shapes}
        output_info = {"names": output_names, "shapes": output_shapes}
        return {"input": input_info, "output": output_info}

    def _run_drpai_translate(
        self,
        model_dir_name: str = "drp_convert_result",
        prepost_yml_path: str = "output.yaml",
        address_yml_path: str = "addr_map.yaml",
    ):
        """Run DRP-AI-Translator with a YAML file"""
        cur_path = os.getcwd()  # get current path
        print(cur_path + "/" + prepost_yml_path)
        self.drp_tran.run_translate(
            model_dir_name,
            onnx=self.model_path,
            CheckPrePost=True,
            prepost=cur_path + "/" + prepost_yml_path,
            addr=cur_path + "/" + address_yml_path,
        )
        
    def _convert_NCHW_to_NHWC(self, nchw: List[int]) -> Tuple[List[int], str]:
        """convert NCHW to NHWC"""
        if len(nchw) == 4:
            order = "HWC"
            nhwc = [nchw[2], nchw[3], nchw[1]]
        elif len(nchw) == 3:
            order = "HWC"
            nhwc = [nchw[1], nchw[2], nchw[0]]
        elif len(nchw) == 2:
            order = "HW"
            nhwc = [nchw[1], nchw[0]]
        elif len(nchw) == 1:
            nhwc = [nchw[0]]
            order = "C"
        else:
            raise ConversionError("input dimension <=4 ")
        
        return nhwc, order
        
        
    def _set_drpai_input(self, input_shape):
        """Set input of DRP-AI
        Args:
            input_type (str): Input type for the DRP-AI model. Should be either 'camera' or 'image'.
        """
        
        input_shape, order = self._convert_NCHW_to_NHWC(input_shape)
        
        self.drp_input_name = "image"
        self.drp_prepost.set_input_to_pre(
            self.drp_input_name,
            shape=tuple(input_shape),
            order=order,
            type="uint8",
            format="RGB",
        )

    def _set_drpai_preprocess(self):
        """set up pre-processing in drp-ai"""
        
        input_pre_sequence = list()
        # Add Pre-Processing for each input
        onnx_input_info = self.onnx_io_info["input"]
        for input_name, input_shape in zip(onnx_input_info["names"], onnx_input_info["shapes"]):
            
            input_shape, order = self._convert_NCHW_to_NHWC(input_shape)
            
            # Add pre-processing to cast_fp16 and nomalize
            input_pre_sequence.extend(
                [drp_ai.cast_any_to_fp16()]
            )

            # Set drp_prepost processing sequence
            self.drp_prepost.set_preprocess_sequence(
                src=[self.drp_input_name], dest=[input_name], pp_seq=input_pre_sequence
            )

            # Set input to receive pre-process data
            self.drp_prepost.set_input_to_body(
                input_name,
                shape=tuple(input_shape),
                order=order,
                type="fp16",
                format="RGB",
            )

    def _set_drpai_postprocess(self):
        """set up post-processing in drp-ai"""
        
        onnx_output_info = self.onnx_io_info["output"]
        post_memcopy = drp_ai.memcopy(WORD_SIZE=2) # FP16
        post_sequence = [post_memcopy]		

        # Add post-processing to cast_fp32 for each input
        for output_name, output_shape in zip(onnx_output_info["names"], onnx_output_info["shapes"]):
            post_output_name = "post_" + output_name
            
            output_shape, order = self._convert_NCHW_to_NHWC(output_shape)

            # Set drp_prepost processing sequence
            self.drp_prepost.set_postprocess_sequence(
                src=[output_name], dest=[post_output_name], pp_seq=post_sequence
            )

            # Set output that requires post-process
            self.drp_prepost.set_output_from_body(output_name, shape=tuple(output_shape), order=order, type="fp16")
            # Set output to receive post-process data
            self.drp_prepost.set_output_from_post(post_output_name, shape=tuple(output_shape), order=order, type="fp16")

    def _save_drpai_prepost_yaml(self, file_path: str = "output.yaml"):
        """Save the pre/post information of the DRP-AI you set as a yaml file"""
        self.drp_prepost.save(file_path)

    def _save_drpai_address_yaml(self, target_device: str, file_path: str = "addr_map.yaml"):
        """Save address yaml file for target_device
        Args:
            target_device (str): Target DRP-AI device for the conversion. Should be one of 'v2m', 'v2ma', 'v2l'.
        """

        valid_devices = ["v2m", "v2l", "v2ma", "avnet_v2l"]

        if target_device not in valid_devices:
            raise ConversionError(
                "Invalid target_device parameter. It should be one of 'v2m', 'v2ma', 'v2l', 'avnet_v2l"
            )

        address = None
        if target_device.lower() == "v2m":
            address = "0xC0000000"
        elif target_device.lower() == "v2ma":
            address = "0x40000000"
        elif target_device.lower() == "v2l" or "avnet_v2l":
            address = "0x80000000"

        self.drp_tran.make_addrfile(address, addr_file=file_path)

class ConversionError(Exception):
    pass

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=True)
    args = parser.add_argument_group("Options")

    args.add_argument(
        "-m",
        "--onnx_model_path",
        dest="onnx_model_path",
        required=True,
        type=str,
        default=None,
        help="onnx_model is required as input.",
    )

    args.add_argument(
        "-t",
        "--target_device",
        dest="target_device",
        type=str,
        default="v2l",
        help="target_device is required as input.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    onnx_path = args.onnx_model_path
    target_device = args.target_device
    
    convert = DRPAIConverter(onnx_path)
    convert.convert_onnx_to_drpai(target_device)
    
if __name__ == "__main__":
    main()
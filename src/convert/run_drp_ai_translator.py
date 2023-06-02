"""
Latency 측정 용 모델 변환 스크립트
모든 모델 변환 가능
실행 명령어 예시
python3 run_drp_ai_translator.py {절대경로/모델.onnx} {타겟 디바이스 : v2m, v2ma, v2l}
"""

import sys
import os
import onnxruntime as ort

TRANSLATOR=os.getenv("TRANSLATOR")
if(TRANSLATOR == None):
    print("[Error] No environment variable")
    print("        Before running this script,")
    print("        Please set environment variable(TRANSLATOR)")
    print("        to the directory where you installed DRP-AI Translator")
    print("        e.g. $export TRANSLATOR=/home/user/drp-ai_translator_release/")
    sys.exit(-1)
DRP_TOOLS = TRANSLATOR + "DRP-AI_translator"
sys.path.append(DRP_TOOLS)
from python_api import *

if __name__ == '__main__':
    """
    * This script & APIs are beta version. 
      The specifications are subject to update.
    This is a sample script to run DRP-AI Translator.
    Before calling DRP-AI translator, 
    generate preposet definition file(.yaml) by python APIs.
    Please refer to the User's manual for the details of the arguments.
    [Note] Before running this script,
           Please set environment variable(TRANSLATOR)
           to the directory where you installed DRP-AI Translator
           e.g. $export TRANSLATOR=/home/user/drp-ai_translator_release/
    """
    cur_path = os.getcwd() # get current path
    onnx_model = sys.argv[1]
    target_device = sys.argv[2]
    output_dir_name = onnx_model.split('/')[-1]

    profile_session = ort.InferenceSession(path_or_bytes=onnx_model)
    input_node = profile_session.get_inputs()[0].name	
    	
    input_shape = profile_session.get_inputs()[0].shape	
    input_shape.append(input_shape.pop(1))  # NCHW -> NHWC로 변경하는 과정	
    input_channel = input_shape[3]		
    print("#### information ###")
    print('model name : ' +str(onnx_model))	
    print('input shape : ' +str(input_shape))	
    print('input shape : ' +str(input_shape[1:]))	
    print('input resize : ' +str(input_shape[1:3]))	
    print('input channel : ' +str(input_channel))	
    print("#### END ###")	

    # [1] Set pre & post processing information
    pp = drp_prepost()

    # camera_data	
    input_name = "image_data"	
    input_channel = 2
    pp.set_input_to_body(input_node, shape=input_shape[1:],order="HWC",type="fp16",format="RGB")	
    # 카메라는 480 640 2만 적용된다. YUV2형식이며, 다른 shape로 바꾸는 것보다 480, 640 형태가 최적화상태
    # Set Post processing sequence
    pp.set_input_to_pre(input_name, shape=[480,640,input_channel],order="HWC",type="uint8",format="YUY2")	
    for i in range(0, len(profile_session.get_outputs())):	
      output_node = profile_session.get_outputs()[i].name	
      post_output_node = 'post_' + output_node	
      output_shape = profile_session.get_outputs()[i].shape	
      output_shape.append(output_shape.pop(1))		
      print('output name : ' +str(output_node))		
      print('output shape : ' +str(output_shape))		
      print('output shape : ' +str(output_shape[1:]))		
      pp.set_output_from_body(output_node, shape=output_shape[1:],order="HWC",type="fp16")		
      pp.set_output_from_post(post_output_node, shape=output_shape[1:],order="HWC",type="fp32")		
      post2_cast_fp16_fp32 = cast_fp16_fp32(CAST_MODE=0)		
      post_sequence = [post2_cast_fp16_fp32]		
      pp.set_postprocess_sequence(src=[output_node],dest=[post_output_node],pp_seq=post_sequence)	
    # [2] Define Pre & Post processing classes	
    pre1_conv_yuv2rgb = conv_yuv2rgb(DOUT_RGB_FORMAT=0)	
    pre2_resize_hwc   = resize_hwc(shape_out=input_shape[1:3],RESIZE_ALG=1)	
    pre3_cast         = cast_any_to_fp16()	
    pre4_normalize    = normalize(cof_add=[0.0, 0.0, 0.0],\
                                cof_mul= [0.00392157, 0.00392157, 0.00392157])	
    # [3] Set Pre processing sequence	
    # post processing은 위에서 설정했음	
    pre_sequence  = [pre1_conv_yuv2rgb,pre2_resize_hwc,pre3_cast,pre4_normalize]

    # [4] Set to drp_prepost processing sequence
    pp.set_preprocess_sequence(src=[input_name],dest=[input_node],pp_seq=pre_sequence)

    # [5] show setting info
    pp.show_params()

    # [6] Save data as yaml file
    pp.save("output.yaml")

    # [7] Initialize DRP-AI Transaltor class
    drp_tran = drp_ai_translator()

    # [8] Choose device of run script
    drp_tran.set_translator("V2M")

    print(target_device)
    # [9] Make address map file
    if target_device == "v2m":
      print('#1')
      drp_tran.make_addrfile("0xC0000000",addr_file="./addr_map.yaml") 
    elif target_device == "v2ma":
      print('#2')
      drp_tran.make_addrfile("0x40000000",addr_file="./addr_map.yaml")
    elif target_device == "v2l":
      print('#3')
      drp_tran.make_addrfile("0x80000000",addr_file="./addr_map.yaml") 
    else:
      print('#3')
      drp_tran.make_addrfile("0xC0000000",addr_file="./addr_map.yaml") 

    # [10] Set onnx model file
    ONNX_MODEL=onnx_model

    # [11] Run DRP-AI Transaltor
    # Set absolute path for input files
    
    translator_path = os.environ["TRANSLATOR"]
    print("Translator path : ",translator_path)
    drp_tran.run_translate(output_dir_name,\
                        onnx= ONNX_MODEL,\
                        prepost=cur_path+"/output.yaml",\
                        addr=cur_path+"/addr_map.yaml"
    )

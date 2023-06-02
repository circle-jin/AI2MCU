# AI2MCU
1. 하나의 모델을 알아서 잘라서 모델 변환까지 하고 싶은 경우

cd /root/AI2MCU
example.json 파일을 수정 후
python3 main.py -c ./json/example.json 

2. 임의의 모델 하나를 변환 시도 하고 싶은 경우
python3 run_drp_ai_translator_test.py /root/yolov5s_NPNet1.onnx v2l

3. 다수의 모델을 자동으로 변환 시도 하고 싶은 경우
/root/AI2MCU/output 폴더에 모델을 넣고, example.json 파일에서 use_crop==false로 설정한 뒤
python3 main.py -c ./json/example.json 


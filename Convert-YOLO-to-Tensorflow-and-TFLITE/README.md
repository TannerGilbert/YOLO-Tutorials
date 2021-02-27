# Convert YOLO to Tensorflow and TFLITE

Darknet Yolo models can be used with Tensorflow, Tensorflow LITE and TensorRT using the [tensorflow-yolov4-tflite repository](https://github.com/hunglc007/tensorflow-yolov4-tflite).

## Setup

```
git clone https://github.com/hunglc007/tensorflow-yolov4-tflite
cd tensorflow-yolov4-tflite
pip install -r requirements.txt # or pip install -r requirements-gpu.txt
```

## Convert to Tensorflow

```
# Convert darknet weights to tensorflow
python save_model.py --weights <path to weights> --output <output> --input_size <input-size> --model <model type (either yolov4 or yolov3)> 

# Run demo tensorflow
python detect.py --weights <path to weights (output from save_model.py)> --size <input-size> --model <model type (either yolov4 or yolov3)>  --image <path to image>
```

## Convert to Tensorflow Lite

```
# Save tf model for tflite converting
python save_model.py --weights <path to weights> --output <output> --input_size <input-size> --model <model type (either yolov4 or yolov3)> --framework tflite

# Create tflite model
python convert_tflite.py --weights <path to weights (output from save_model.py)> --output <tflite output>

## Quantize float16
python convert_tflite.py --weights <path to weights (output from save_model.py)> --output <tflite output> --quantize_mode float16

## Quantize int8
python convert_tflite.py --weights <path to weights (output from save_model.py)> --output <tflite output> --quantize_mode int8 --dataset ./coco_dataset/coco/val207.txt

# Run demo tflite model
python detect.py --weights <path to weights> --size <input-size> --model <model type (either yolov4 or yolov3)>  --image <path to image> --framework tflite
```


## Convert to TensorRT

```
python save_model.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf --input_size 416 --model yolov3
python convert_trt.py --weights ./checkpoints/yolov3.tf --quantize_mode float16 --output ./checkpoints/yolov3-trt-fp16-416

# yolov3-tiny
python save_model.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --input_size 416 --tiny
python convert_trt.py --weights ./checkpoints/yolov3-tiny.tf --quantize_mode float16 --output ./checkpoints/yolov3-tiny-trt-fp16-416

# yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4.tf --input_size 416 --model yolov4
python convert_trt.py --weights ./checkpoints/yolov4.tf --quantize_mode float16 --output ./checkpoints/yolov4-trt-fp16-416
```
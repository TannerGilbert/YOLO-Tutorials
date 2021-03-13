# YOLO Object Detection in PyTorch

![](doc/prediction_example.jpg)

## Installation

[yolov3](https://github.com/ultralytics/yolov3) can be installed by cloning the repository and installing the dependencies located inside the [requirements.txt file](https://github.com/ultralytics/yolov3/blob/master/requirements.txt).

```bash
git clone https://github.com/ultralytics/yolov3
cd yolov3
pip install -qr requirements.txt
```

## Detection Using A Pre-Trained Model

You can run an object detection model using the ```detect.py``` file. You can find a list of all the arguments you can parse to ```detect.py``` by specifying the --help flag.

```bash
usage: detect.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--source SOURCE]
                 [--img-size IMG_SIZE] [--conf-thres CONF_THRES]
                 [--iou-thres IOU_THRES] [--device DEVICE] [--view-img]
                 [--save-txt] [--save-conf] [--classes CLASSES [CLASSES ...]]
                 [--agnostic-nms] [--augment] [--update] [--project PROJECT]
                 [--name NAME] [--exist-ok]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS [WEIGHTS ...]
                        model.pt path(s)
  --source SOURCE       source
  --img-size IMG_SIZE   inference size (pixels)
  --conf-thres CONF_THRES
                        object confidence threshold
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --view-img            display results
  --save-txt            save results to *.txt
  --save-conf           save confidences in --save-txt labels
  --classes CLASSES [CLASSES ...]
                        filter by class: --class 0, or --class 0 2 3
  --agnostic-nms        class-agnostic NMS
  --augment             augmented inference
  --update              update all models
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
  --exist-ok            existing project/name ok, do not increment
```

The source could be an image, video, directory of images, webcam or an image stream.

* Image: --source file.jpg
* Video: --source file.mp4
* Directory: --source dir/
* Webcam: --source 0
* RTSP stream: --source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa
* HTTP stream: --source http://wmccpinetop.axiscam.net/mjpg/video.mjpg

If you don't specify a source, it uses the data/images folder. The results will automatically be saved inside the runs/detect folder.

```
!python detect.py --weights yolov3.pt --img 640 --conf 0.25 --source data/images/
```

![](doc/prediction_example.jpg)


```
python3 detect.py --weights yolov3.pt --source TownCentreXVID.avi
```

[![pedestrian detection](https://img.youtube.com/vi/9Mdc-HU6BV8/0.jpg)](https://www.youtube.com/watch?v=9Mdc-HU6BV8)

## Train on custom data

### 1. Create annotations

After collecting your images, you'll have to annotate them. For YOLO, each image should have a corresponding .txt file with a line for each ground truth object in the image that looks like:

```bash
<object-class> <x> <y> <width> <height>
```

The .txt file should have the same name as the image. All images should be located inside a folder called images and all labels should be located inside the labels folder. 

You can get such labels using an annotation tool like [labelImg](https://github.com/tzutalin/labelImg), which supports both Pascal VOC and YOLO (just make sure that you have selected YOLO).

![](doc/labelImg.PNG)

If you have a dataset with PASCAL VOC labels you can convert them using the [```convert_voc_to_yolo.py``` script](convert_voc_to_yolo.py). Before you execute the file you'll have to change the classes list to fit your dataset. After that you can run the script:

```bash
python convert_voc_to_yolo.py
```

### 2. Create dataset.yaml file

The dataset.yaml file defines defines 1) an optional download command/URL for auto-downloading, 2) a path to a directory of training images (or path to a *.txt file with a list of training images), 3) the same for our validation images, 4) the number of classes, 5) a list of class names:

microcontroller-detection.yml:
```yml
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: microcontroller-detection/train.txt
val: microcontroller-detection/train.txt

# number of classes
nc: 4

# class names
names: ['Arduino_Nano', 'Heltec_ESP32_Lora', 'ESP8266', 'Raspberry_Pi_3']
```

### 3. Start Training

To train the model pass your yml file to the `train.py` script. You can also pass additional arguments like the image size, batch size and epoch count. If you want to start from a pretrained model (recommended) you also need to specify the `--weights` argument (Pretrained weights are auto-downloaded from the [latest YOLOv3 release](https://github.com/ultralytics/yolov3/releases)). If you want to train from scratch (starting with random weights) you can use `--weights '' --cfg yolov3.yaml`.

```bash
python train.py --img 640 --batch 16 --epochs 300 --data microcontroller-detection.yml --weights yolov3.pt
```

![](doc/start_training.PNG)
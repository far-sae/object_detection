
# Object Detection System Using SSD Mobilenet V2

This repository contains an Object Detection system based on the SSD Mobilenet V2 model. The system processes images and detects objects using a pre-trained TensorFlow model.

## Features

- **Object Detection**: The `object.py` script processes images and detects objects using the SSD Mobilenet V2 model.
- **Customizable**: The configuration file `ssd_mobilenet_v2_coco_2018_03_29.pbtxt` allows you to modify the detection parameters.

## Installation

To set up the environment, clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/far-sae/object_detection.git
cd oobject_detection
pip install -r requirements.txt
```

## Download the Model

You need to download the pre-trained model `frozen_inference_graph.pb` to run the object detection script.

1. Download the model from the TensorFlow Model Zoo: [frozen_inference_graph.pb](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz).
2. Extract the `frozen_inference_graph.pb` file and place it in the project directory.

## Usage

### Object Detection

Use the `object.py` script to perform object detection on images.

```bash
python object.py --image /path/to/your/image --graph frozen_inference_graph.pb --config ssd_mobilenet_v2_coco_2018_03_29.pbtxt
```

## Files and Directories

- **`object.py`**: Script for detecting objects in images using SSD Mobilenet V2.
- **`ssd_mobilenet_v2_coco_2018_03_29.pbtxt`**: Configuration file for the SSD Mobilenet V2 model.
- **`requirements.txt`**: File listing the required Python packages.
- **`frozen_inference_graph.pb`**: Pre-trained TensorFlow model that needs to be downloaded.

## Dependencies

The project relies on several Python libraries, which are listed in the `requirements.txt` file. You can install these dependencies using the command mentioned above.

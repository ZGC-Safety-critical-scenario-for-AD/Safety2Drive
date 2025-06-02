# Adversarial Patches against UAE Object Detection

## Setup

Tested with python 3.8

```shell
python -m venv venv
source venv/bin/activate
pip install tqdm==4.66.4
pip install imagesize==1.4.1
pip install opencv-python==4.9.0.80
```

## VisDrone Dataset Format

Dataset can be downloaded from <https://github.com/VisDrone/VisDrone-Dataset>.

Annotations for the detections are follows:

`<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>`

```
 -   <bbox_left> The x coordinate of the top-left corner of the predicted bounding box
 -   <bbox_top> The y coordinate of the top-left corner of the predicted object bounding box
 -   <bbox_width> The width in pixels of the predicted object bounding box
 -   <bbox_height> The height in pixels of the predicted object bounding box
 -   <score> The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing an object instance. The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, while 0 indicates the bounding box will be ignored.
 -   <object_category> The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), others(11))
 -   <truncation> The score in the DETECTION result file should be set to the constant -1.The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
 -   <occlusion> The score in the DETECTION file should be set to the constant -1. The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).
```

## Download VisDrone Dataset

Download Dataset from <https://github.com/VisDrone/VisDrone-Dataset> and unzip and place under top level directory `data`.

Alternatively, use gdown to download the zip files from the command line.

```shell
mkdir data
cd data
pip install gdown
# object detection train subset
gdown 1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn
# object detection val subset
gdown 1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59
# object detection test subset
gdown 1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V

# unzip all data
unzip VisDrone2019-DET-test-dev.zip
unzip VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-val.zip

# remove all zip files
rm *.zip
```

## Visualize Images

Running annotation visualizations for VisDrone or YOLO format.

```shell
python disp_visdrone.py -a ANNOTS_DIR -i IMAGES_DIR
python disp_yolo.py -a ANNOTS_DIR -i IMAGES_DIR
```

## Convert VisDrone to YOLO annotation format

Note: The classes to consider and any additional re-assingment of classes must be done with variables `CLASS_2_CONSIDER` and `CLASS_ID_REMAP` inside `conv_visdrone_2_yolo_fmt.py`.

Can use optional params `low_dim_cutoff` and `low_area_cutoff` to cutoff bunding boxes that do not satisfy a minimum box dimension or area percentage cutoff.

```shell
# example conversion of VisDrone train, val and test set annotations to YOLO format. Use -h for all options
python conv_visdrone_2_yolo_fmt.py --sad data/VisDrone2019-DET-train/annotations --sid data/VisDrone2019-DET-train/images --td data/VisDrone2019-DET-train/labels
python conv_visdrone_2_yolo_fmt.py --sad data/VisDrone2019-DET-val/annotations --sid data/VisDrone2019-DET-val/images --td data/VisDrone2019-DET-val/labels
python conv_visdrone_2_yolo_fmt.py --sad data/VisDrone2019-DET-test-dev/annotations --sid data/VisDrone2019-DET-test-dev/images --td data/VisDrone2019-DET-test-dev/labels
```

Note: To convert to COCO annotation format refer to <https://github.com/SamSamhuns/ml_data_processing/tree/master/annotation_format_conv>

## References

- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)

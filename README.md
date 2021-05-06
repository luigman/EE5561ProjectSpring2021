# EE5561ProjectSpring2021

A PyTorch implementation of UNet semantic segmentation network, applied to the SpaceNet v1 Building Detection Challenge

## Contributors
- Liam Coulter
- Teague Hall
- Luis Guzman
- Isaac Kasahara

## Instructions

Download the [training dataset images](https://drive.google.com/drive/folders/1tRGBqkVs1ZvcJhOrKFhK_NQdiKP1qOvg?usp=sharing) (~3.71 GB), and [image masks](https://drive.google.com/drive/folders/1xhcLVDd6sknj-x2hzsy_HRqWlmsPmXhV?usp=sharing) (~35.7 MB)

Optionally download the [geojson vector label data](https://drive.google.com/drive/folders/1sBP13AUGu73PP26Mxve1_2nuxUoprsVn?usp=sharing) (~201 MB) to test preprocessing scripts

Place the training images in `SN1_buildings_train_AOI_1_Rio_3band/3band/*.tif` and the masks in `building_mask/*.tif`

To test a pre-trained model, download the [network weights](https://drive.google.com/file/d/1nRcyDQbID962SVvL6JL1_7yoehOHEl3p/view?usp=sharing) and place it in the `trained_models` folder.

Official data for SpaceNet v1 Building Challenge is also accessible through [AWS](https://registry.opendata.aws/spacenet/).

## Training

To train the UNet, run `train.py` using the command `python train.py --batch_size 16`

A batch size of 16 takes approximately 12GB of video memory, so adjust the batch size according to what you have availible

## Testing

To view the results, run `test.py`

The pre-trained model can be downloaded [here](https://drive.google.com/file/d/1nRcyDQbID962SVvL6JL1_7yoehOHEl3p/view?usp=sharing) and should be placed in a directory called `trained_models`.

## Evaluation

Network evaluation can be calculated with `evaluate.py`

## Preprocessing
Note: We provide already-processed [image masks](https://drive.google.com/drive/folders/1xhcLVDd6sknj-x2hzsy_HRqWlmsPmXhV?usp=sharing), so the following step is not necessary to test our implementation.

To run `image_json_to_mask.py`, you need to first install the SpaceNet Utilities from [here](https://github.com/SpaceNetChallenge/utilities). Follow the instructions and download necessary files. It worked for Liam on MacOS to make a Conda virtual environment, and load everything in there. The only package I had trouble with was `rasterio`, since I assume it is older. Note that the file `geoTools.py` at the base repository level is a modified version of `geoTools.py` that comes with the SpaceNet utilities install, and includes 1 extra function (`latlonToPixel`). 

Once you have the SpaceNet utilities, edit directory variables in `image_json_to_mask.py` to reflect your setup, and you should be able to run.

## Acknowledgments
- [Adam Van Etten](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53)

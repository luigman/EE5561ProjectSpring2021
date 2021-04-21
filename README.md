# EE5561ProjectSpring2021

## Contributors
- Liam Coulter
- Teague Hall
- Luis Guzman
- Isaac Kasahara

## Data Access
Data for SpaceNet v2 Building Challenge is accessible through [AWS](https://registry.opendata.aws/spacenet/). I have also uploaded the training dataset images to my [Google Drive](https://drive.google.com/drive/folders/1tRGBqkVs1ZvcJhOrKFhK_NQdiKP1qOvg?usp=sharing) (~3.71 GB), along with the [geojson vector label data](https://drive.google.com/drive/folders/1sBP13AUGu73PP26Mxve1_2nuxUoprsVn?usp=sharing) (~201 MB). 

## Image Masks
To run `image_json_to_mask.py`, you need to first install the SpaceNet Utilities from [here](https://github.com/SpaceNetChallenge/utilities). Follow the instructions and download necessary files. It worked for Liam on MacOS to make a Conda virtual environment, and load everything in there. The only package I had trouble with was `rasterio`, since I assume it is older. Note that the file `geoTools.py` at the base repository level is a modified version of `geoTools.py` that comes with the SpaceNet utilities install, and includes 1 extra function (`latlonToPixel`). 

Once you have the SpaceNet utilities, edit directory variables in `image_json_to_mask.py` to reflect your setup, and you should be able to run.

Image masks can be found on my [Google Drive](https://drive.google.com/drive/folders/1xhcLVDd6sknj-x2hzsy_HRqWlmsPmXhV?usp=sharing). The folder is ~40 MB and consists of `.tif` binary mask images for each training image in the SpaceNet V2 challenge. 

## Acknowledgments
- [Adam Van Etten](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53)
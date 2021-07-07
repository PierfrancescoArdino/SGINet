# SGINet

Pytorch implementation of our paper [Semantic-Guided Inpainting Network for Complex UrbanScenes Manipulation](https://ieeexplore.ieee.org/abstract/document/9412690/)
In [ICPR 2020](https://www.micc.unifi.it/icpr2020/).
Please cite with the following Bibtex code:
```
@INPROCEEDINGS{9412690,
  author={Ardino, Pierfrancesco and Liu, Yahui and Ricci, Elisa and Lepri, Bruno and de Nadai, Marco},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  title={Semantic-Guided Inpainting Network for Complex Urban Scenes Manipulation}, 
  year={2021},
  volume={},
  number={},
  pages={9280-9287},
  doi={10.1109/ICPR48806.2021.9412690}}
```

Please follow the instructions to run the code.

# Scripts

## 1. Installation

 - See the [`sgi_net.yml`](./sgi_net.yml) configuration file. We provide an user-friendly configuring method via Conda system, and you can create a new Conda environment using the command:

```
conda env create -f sgi_net.yml
conda activate sgi_net
```

 - Install `cityscapesscripts` with `pip`
```
cd cityscapesScripts
pip install -e .
```
 - Install `nvidia-apex` using
```
sh install_nvidia_apex.sh
```

## 2. Data Preprocessing
### Cityscapes
* Please download the Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/) (registration required). After downloading, please put these files under the ```~/datasets/cityscapes/``` folder and run the following command in order to generate the correct segmentation maps
  ```
  cd cityscapesScripts
  CITYSCAPES_DATASET=~/datasets/cityscapes/
  python cityscapesscripts/preparation/createTrainIdLabelImgs.py
  ```
  If you want to use a different number of labels for the segmentation you can change them in the ```cityscapesScripts/cityscapesscripts/helpers/labels.py``` file. 

  You should end up with the following structure:
  ```
  datasets
  ├── cityscapes
  │   ├── leftImg8bit_sequence
  │   │   ├── train
  │   │   │   ├── aachen
  │   │   │   │   ├── aachen_000003_000019_leftImg8bit.png
  │   │   │   │   ├── ...
  │   │   ├── val
  │   │   │   ├── frankfurt
  │   │   │   │   ├── frankfurt_000000_000294_leftImg8bit.png
  │   │   │   │   ├── ...
  │   ├── gtFine
  │   │   ├── train
  │   │   │   ├── aachen
  │   │   │   │   ├── aachen_000003_000019_gtFine_trainIds.png
  │   │   │   │   ├── aachen_000003_000019_gtFine_polygons.json
  │   │   │   │   ├── aachen_000003_000019_gtFine_instanceIds.png
  │   │   │   │   ├── ...
  │   │   ├── val
  │   │   │   ├── frankfurt
  │   │   │   │   ├── frankfurt_000000_000294_gtFine_trainIds.png
  │   │   │   │   ├── frankfurt_000000_000294_gtFine_polygons.json
  │   │   │   │   ├── frankfurt_000000_000294_gtFine_instanceIds.png
  │   │   │   │   ├── ...
  ```
* Then run the script ```src/preprocess_city.py``` in order to prepare the dataset.
  #### Usage
    The script takes as input three parameters:
     - `dataroot`: Folder where the Cityscape dataset has been extracted.
     - `resize_size`: New size of the images (width,height). By default the images will not be resized. Default value: (2048,1024)
     - `use_multiprocessing`: Run the preprocessing in parallel. By default is disabled
  #### Example  
   ```
  cd src
  python preprocess_city.py --dataroot ~/datasets/cityscapes/ --resize_size 512,256 --use_multiprocessing
  ```
* Copy the train list and the evaluation list from ```file_list/cityscapes``` into the dataroot ```~/datasets/cityscapes/```
  ```bash
  cp file_list/cityscapes/* ~/dataset/cityscapes/
   ```
    You should end up with the following structure:
  ```
  datasets
  ├── cityscapes
  │   ├── train_img
  │   │   ├── aachen_000003_000019_leftImg8bit.png
  │   │   ├── ...
  │   ├── val_img
  │   │   ├── frankfurt_000000_000294_leftImg8bit.png
  │   │   ├── ...
  │   ├── train_label
  │   │   ├── aachen_000003_000019_gtFine_trainIds.png
  │   │   ├── ...
  │   ├── val_label
  │   │   ├── frankfurt_000000_000294_gtFine_trainIds.png
  │   │   ├── ...
  │   ├── train_inst
  │   │   ├── aachen_000003_000019_gtFine_data.json
  │   │   ├── aachen_000003_000019_gtFine_instanceIds.png
  │   │   ├── ...
  │   ├── val_inst
  │   │   ├── frankfurt_000000_000294_gtFine_data.json
  │   │   ├── frankfurt_000000_000294_gtFine_instanceIds.png
  │   │   ├── ...
  │   ├── train.txt
  │   ├── val.txt
  ```
#### Indian Driving Dataset
TODO
### 3. Train
#### Single-GPU Train
* Train a model at 256 x 256 resolution with cropping and Pixel Shuffle in the decoder
  ```bash
  cd src
  sh script/train_paper_cityscapes_pixel_shuffle.sh
  ```
* Train a model at 256 x 256 resolution with cropping and deconvolution in the decoder
  ```bash
  cd src
  sh script/train_paper_cityscapes.sh
  ```
#### Multi-GPU Train
* Train a model at 256 x 256 resolution with cropping and Pixel Shuffle in the decoder
  ```bash
  cd src
  sh script/train_paper_cityscapes_multigpu_pixel_shuffle.sh
  ```
* Train a model at 256 x 256 resolution with cropping and deconvolution in the decoder
  ```bash
  cd src
  sh script/train_paper_cityscapes_multigpu.sh
  ```
The example consider a scenario with a single node and two gpus per node. Please change according to your needs. For more information check the [DDP example](https://github.com/pytorch/examples/tree/master/distributed/ddp)

#### Training with Automatic Mixed Precision (AMP) for faster speed
* Train a model at 256 x 256 resolution with cropping and Pixel Shuffle in the decoder
  ```bash
  cd src
  sh script/train_paper_cityscapes_fp16_multigpu_pixel_shuffle.sh
  ```
* Train a model at 256 x 256 resolution with cropping and deconvolution in the decoder
  ```bash
  cd src
  sh script/train_paper_cityscapes_fp16_multigpu.sh
  ```
The example consider a scenario with a single node and two gpus per node. Please change according to your needs. For more information check the [DDP example](https://github.com/pytorch/examples/tree/master/distributed/ddp)
### 4. Test
WORK IN PROGRESS

## More Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
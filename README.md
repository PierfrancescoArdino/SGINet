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

## Installation

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

## Data Preprocessing
### Cityscapes
* Please download the Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/) (registration required). After downloading, please put these files under the ```~/datasets/cityscapes/``` folder and run the following command in order to generate the correct segmentation maps
  ```
  cd cityscapesScripts
  CITYSCAPES_DATASET=~/datasets/cityscapes/ python cityscapesscripts/preparation/createTrainIdLabelImgs.py
  ```
  If you want to use a different number of labels for the segmentation you can change them in the ```cityscapesScripts/cityscapesscripts/helpers/labels.py``` file.
  
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

#### Indian Driving Dataset
TODO

WORK IN PROGRESS
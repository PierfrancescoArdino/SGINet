from .cityscapes import CityscapesSegmentation
from .indianDrivingDataset import IndianDrivingDataset


datasets = {
    'cityscapes': CityscapesSegmentation,
    'indiandrivingdataset': IndianDrivingDataset,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)

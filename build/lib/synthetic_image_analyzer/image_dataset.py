import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image,ImageOps


class image_dataset(Dataset):

    def __init__(self,filepath,
                 class_names = None,
                 num_channels=1,resolution=1024):  # crop_size,
        """
        Define the dataset for feature extraction. The dataset should be arranged as follows:
        ├── real_dataset
        |   ├── class1
        |   ├── class2
        |   ├── class3
        |   └── ...
        ├── synthetic_dataset1
        |   ├── class1
        |   ├── class2
        |   ├── class3
        |   └── ...
        ├── synthetic_dataset2
        |   ├── class1
        |   ├── class2
        |   ├── class3
        |   └── ...

        Parameters:
        - class_names: A list with the classes to be calculated. Default is None, which calculates
        all classes.
        - num_channels: The number of channels for input images, only 1 and 3 is accepted for pre-trained models.
        - resolution: The reslution for input images.

        Returns:
        float: The dataset contains all images from the filepath.
        """

        self.imlist = []

        self.filepath = filepath
        if class_names is None:
            class_names = os.listdir(self.filepath)
        for label in class_names:
            newpath = os.path.join(self.filepath,label)
            for file in os.listdir(newpath):
                self.imlist.append(os.path.join(self.filepath,label,file))

        self.class_names = class_names
        self.transforms = transforms
        self.num_class = len(class_names)

        self.num_channels = num_channels

        self.transforms =  transforms.Compose([

            transforms.Resize([resolution,resolution]),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * num_channels, [0.5] * num_channels)
        ])
    def __getitem__(self, idx):
        if idx < len(self.imlist):
            img_name = self.imlist[idx]

            img = Image.open(img_name)
            if self.num_channels ==1:
                img = ImageOps.grayscale(img)
            img = self.transforms(img)

            return {'image':img,'filename':img_name,}


    def __len__(self):
        return len(self.imlist)
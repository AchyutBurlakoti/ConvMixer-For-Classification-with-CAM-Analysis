import glob
import random

from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

labels_mapping = {"Uninfected": 0, "Parasitized": 1}

def get_transform(h, w):
    train_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=12),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])

    return train_transform, test_transform

class ImageDataset(Dataset):

    def __init__(self, status = 'train', h=64, w=64):
        super().__init__()

        self.status = status

        if status == 'train':
            self.data = glob.glob('./archive/train/*/*.png')
            self.transform = transforms.Compose([
                                transforms.Resize((h, w)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandAugment(num_ops=2, magnitude=12),
                                transforms.ColorJitter(0.2, 0.2, 0.2),
                                transforms.ToTensor(),
                                transforms.RandomErasing(p=0.2)
                            ])
        else :
            self.data = glob.glob('./archive/test/*/*.png')
            self.transform = transforms.Compose([
                                transforms.Resize((h, w)),
                                transforms.ToTensor()
                            ])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        
        filename = self.data[index]
        labelstr = filename.split('/')[-1].split('\\')[-2]
        label = labels_mapping[labelstr]

        image = Image.open(filename)
        output = self.transform(image)
        
        return output, label
    
if __name__ == '__main__':
    dataset = ImageDataset('test')
    print(dataset.__len__())
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset,DataLoader
from torchvision import transforms
import cv2
import numpy as np

def load_dataset(val_size=0.2):
    #load dataset from MNIST dataset
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    train_indices, val_indices, _, _ = train_test_split(
        range(len(training_data)),
        training_data.targets,
        stratify=training_data.targets,
        test_size=val_size,
    )
    
    # generate subset based on indices
    train_data = Subset(training_data, train_indices)
    val_data = Subset(training_data, val_indices)
    
    return train_data, val_data, test_data

#canny edge detection transform 
class CannyEdge:
    def __init__(self):
        pass
        #super().__init__()
    
    def forward(self,img):
        img=np.array(img.permute(1,2,0))
        img=(img*255).astype(np.uint8)
        edges = cv2.Canny(img,50,150)
        edges=(edges/255).astype(np.float32)
        torch.tensor(edges).unsqueeze(0)
        return edges
    
    def __call__(self, sample):
        return self.forward(sample)

class Contraversion:
    def __init__(self):
        pass
        #super().__init__()
    
    def forward(self,img):
        img=np.array(img.permute(1,2,0))
        img2=img.copy()
        img2=1-img2
        #img=torch.tensor(img2)
        return img2
    
    def __call__(self, sample):
        return self.forward(sample)
    
class CustomDataset(Dataset):
    
    def __init__(self,dataset, transform=None):    
        self.transform=transform
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)
    
    def canny_transform(self):
        return transforms.Compose([CannyEdge(), ToTensor()])
    
    def contrav_transform(self):
        return transforms.Compose([Contraversion(), ToTensor()])
    
    def __getitem__(self, idx):      
        img,y=self.dataset[idx]
        if self.transform:
            if self.transform=='canny':
                transform=self.canny_transform()
            elif self.transform=='contraversion':
                transform=self.contrav_transform()
            img = transform(img)
        
        return img,y


def get_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size, shuffle=True)
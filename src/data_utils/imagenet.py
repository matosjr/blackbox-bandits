import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

IMAGENET_DATA_LOCATION = 'D:/Workspace/Scripts/datasets/imagenet'

class ImagetNetDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor, transform=None):
        assert data_tensor.shape[0] == labels_tensor.shape[0]
        assert all(data_tensor[0].size == img_t.size for img_t in data_tensor)
        assert all(labels_tensor[0].size == label_t.size for label_t in labels_tensor)

        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor
        self.tranform = transform
        
    def __getitem__(self, index):
        x = self.data_tensor[index]
        x = Image.fromarray(x)

        if self.tranform:
            x = self.tranform(x)
            
        y = self.labels_tensor[index]
        
        return x, y
    
    def __len__(self):
        return self.data_tensor.shape[0]
    
    
def load_imagent_data():
    path = os.path.join(IMAGENET_DATA_LOCATION, 'imagenet_test_data.npy')
    f = np.load(path)
    x_test = f.T.astype('float32')
    x_test = x_test.reshape(x_test.shape[-1], *x_test.shape[:-1][::-1])
    x_test = (x_test + 0.5) * 255
    x_test = np.clip(x_test, a_min = 0, a_max = 255)
    x_test = np.uint8(x_test)
        
    path = os.path.join(IMAGENET_DATA_LOCATION, 'imagenet_test_labels.npy')
    f = np.load(path)
    y_test = f.T.astype('int32')
    y_test = np.argmax(y_test, axis=0)
    # y_test = y_test.reshape(y_test.shape[0], 1)
    return (x_test, y_test)

def imagenet_data(transform=None, batch_size=50):
    x_test, y_test = load_imagent_data()
    test_dataset = ImagetNetDataset(x_test, y_test, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
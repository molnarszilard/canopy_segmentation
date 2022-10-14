import torch.utils.data as data
from pathlib import Path
import os
import cv2
import numpy as np

class DataLoader(data.Dataset):
    
    def __init__(self, root='./dataset', train=True):
        self.root = Path(root)
        if train:
            self.image_input_paths = [root+'/images/train/'+d for d in os.listdir(root+'/images/train')]
        else:
            self.image_input_paths = [root+'/images/test/'+d for d in os.listdir(root+'/images/test/')]        
        self.length = len(self.image_input_paths)
            
    def __getitem__(self, index):
        path = self.image_input_paths[index]
        image_input = cv2.imread(path).astype(np.float32)
        image_input = cv2.resize(image_input,(640,480))
        image_input = np.moveaxis(image_input,-1,0)
        maskgt = cv2.imread(path.replace('images', 'masks')).astype(np.float32)
        maskgt = cv2.resize(maskgt,(640,480))
        maskgt = np.moveaxis(maskgt,-1,0)
        # print(maskgt.shape)
        return image_input/255, maskgt/255

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DataLoader()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())

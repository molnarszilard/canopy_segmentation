import torch.utils.data as data
from pathlib import Path
import os
import cv2
import numpy as np

class DataLoader(data.Dataset):
    
    def __init__(self, root='./dataset', train=True,cs="rgb",img_height=480,img_width=640):
        self.root = Path(root)
        self.cs=cs
        self.height=img_height
        self.width=img_width
        if os.path.exists(root+'/images/train/'):
            if train:
                self.image_input_paths = [root+'/images/train/'+d for d in os.listdir(root+'/images/train') if d.endswith("jpg") or d.endswith("png")]
            else:
                self.image_input_paths = [root+'/images/test/'+d for d in os.listdir(root+'/images/test/') if d.endswith("jpg") or d.endswith("png")]
        elif os.path.exists(root+'/train/'):
            if train:
                self.image_input_paths = [root+'/train/images/'+d for d in os.listdir(root+'/train/images/') if d.endswith("jpg") or d.endswith("png")]
            else:
                self.image_input_paths = [root+'/test/images/'+d for d in os.listdir(root+'/test/images/') if d.endswith("jpg") or d.endswith("png")]
        else:
            print("Given Dataset path is not valid. It must contain the images and the masks organized into train and test directories.")
        self.length = len(self.image_input_paths)
            
    def __getitem__(self, index):
        path = self.image_input_paths[index]
        # print(path)
        if self.cs=="rgb":
            image_input = cv2.imread(path).astype(np.float32)
        elif self.cs=="lab":
            image_input = cv2.imread(path.replace('images', 'images_lab')).astype(np.float32)
            # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2LAB).astype(np.float32)
        elif self.cs=="luv":
            image_input = cv2.imread(path.replace('images', 'images_luv')).astype(np.float32)
            # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2LUV).astype(np.float32)
        elif self.cs=="hls":
            image_input = cv2.imread(path.replace('images', 'images_hls')).astype(np.float32)
            # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2HLS).astype(np.float32)
        elif self.cs=="hsv":
            image_input = cv2.imread(path.replace('images', 'images_hsv')).astype(np.float32)
            # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV).astype(np.float32)
        elif self.cs=="ycrcb":
            image_input = cv2.imread(path.replace('images', 'images_ycrcb')).astype(np.float32)
            # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        else:
            print("Unknown color space.")
        image_input = cv2.resize(image_input,(self.width,self.height))
        image_input = np.moveaxis(image_input,-1,0)
        maskgt = cv2.imread(path.replace('images', 'masks')).astype(np.float32)
        maskgt = cv2.resize(maskgt,(self.width,self.height))
        maskgt = maskgt[:,:,0]
        maskgt = np.expand_dims(maskgt,axis=-1)
        maskgt = np.moveaxis(maskgt,-1,0)
        # print(maskgt.shape)
        # if self.cs=="rgb":
            # image_input/=255
        return image_input, maskgt/255

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DataLoader()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())

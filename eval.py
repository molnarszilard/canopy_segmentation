import argparse
import cv2
import numpy as np
import os
import timeit
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from network import NetworkModule

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Normal image estimation from ToF depth image')
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true', help='whether use CUDA')
    parser.add_argument('--input_folder', dest='input_folder', default='./dataset/input_images/aghi/', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='saved_models/saved_model__1_9.pth', type=str, help='path to the model to use')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: CUDA device is available. You might want to run the program with --cuda=True")
    
    # network initialization
    print('Initializing model...')
    net = NetworkModule(fixed_feature_weights=False)
    if args.cuda:
        net = net.cuda()
    print("Model initialization done.")  
    
    load_name = os.path.join(args.model_path)
    print("loading checkpoint %s" % (load_name))
    state = net.state_dict()
    checkpoint = torch.load(load_name)
    checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
    state.update(checkpoint)
    net.load_state_dict(state)
    if 'pooling_mode' in checkpoint.keys():
        POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
    del checkpoint
    torch.cuda.empty_cache()
    net.eval()

    print('evaluating...')
    with torch.no_grad():
        if args.input_folder.endswith('.png') or args.input_folder.endswith('.png'):
            img = cv2.imread(args.depth_folder).astype(np.float32)
            img = cv2.resize(img,(640,480))
            img = np.moveaxis(img,-1,0)/255
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = img.cuda()
            start = timeit.default_timer()
            maskpred = net(img)
            stop = timeit.default_timer()
            threshold = maskpred.mean()
            imgmasked = img.clone()
            imgmasked[maskpred>=threshold]/=3
            dirname, basename = os.path.split(args.input_folder)
            save_path=args.pred_folder+basename[:-4]
            save_image(imgmasked[0], save_path +"_pred"+'.png')
            print('Predicting the image took ', stop-start)
        else:
            dlist=os.listdir(args.input_folder)
            dlist.sort()
            time_sum = 0
            counter = 0
            for filename in dlist:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    path=args.input_folder+filename
                    print("Predicting for:"+filename)
                    img = cv2.imread(path).astype(np.float32)
                    img = cv2.resize(img,(640,480))
                    img = np.moveaxis(img,-1,0)/255
                    img = torch.from_numpy(img).float().unsqueeze(0)
                    img = img.cuda()
                    start = timeit.default_timer()
                    maskpred = net(img)
                    stop = timeit.default_timer()
                    time_sum=time_sum+stop-start
                    counter=counter+1
                    threshold = maskpred.mean()
                    imgmasked = img.clone()
                    imgmasked[maskpred>=threshold]/=3
                    save_path=args.pred_folder+filename[:-4]
                    save_image(imgmasked[0], save_path +"_pred"+'.png')
                else:
                    continue
            print('Predicting '+str(counter)+' images took ', time_sum/counter)  
    

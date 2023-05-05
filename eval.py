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
    parser = argparse.ArgumentParser(description='canopy segmentation on individual images')
    parser.add_argument('--cuda', dest='cuda', default=True, type=bool, help='whether use CUDA')
    parser.add_argument('--input', dest='input', default='./dataset/input_images/aghi/', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='saved_models/saved_model_1_9.pth', type=str, help='path to the model to use')
    parser.add_argument('--model_size', dest='model_size', default='medium', type=str, help='size of the model: small, medium, large')
    parser.add_argument('--save_type', dest='save_type', default="mask", type=str, help='do you want to save the masked image, the mask, or both: splash, mask, both')
    parser.add_argument('--dim', dest='dim', default=False, type=bool, help='dim the pixels that are not segmented, or leave them black?')
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
    args = parser.parse_args()
    return args

def process_image(directory, filename,counter):
    img = cv2.imread(directory+filename)
    if args.save_type in ['splash','both']:
        imgmasked = img.copy()
        # imgmasked = cv2.resize(imgmasked,(640,480))
        imgmasked = np.moveaxis(imgmasked,-1,0)
        imgmasked = torch.from_numpy(imgmasked).float().unsqueeze(0)
    # if args.cs=="lab":        
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # elif args.cs=="luv":
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV).astype(np.float32)
    # elif args.cs=="hls":
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    # elif args.cs=="hsv":
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)    
    # elif args.cs=="ycrcb":
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    # elif args.cs!="rgb":
    #     print("Unknown color space.")
    #     exit()
    # print(img.shape)
    h,w = img.shape[:2]
    img = cv2.resize(img,(640,480))
    img = np.moveaxis(img,-1,0)
    img = torch.from_numpy(img).float().unsqueeze(0)
    # if args.cs=="rgb":
    # img/=255
    if args.cuda:
        if args.save_type in ['splash','both']:
            imgmasked = imgmasked.cuda()
        img = img.cuda()
    if counter==0:
        start = timeit.default_timer()
        maskpred = net(img) #in order to remove the setup-time
        stop = timeit.default_timer()
        setuptime = stop-start
    else:
        setuptime = 0
    start = timeit.default_timer()
    maskpred = net(img)
    stop = timeit.default_timer()
    threshold = maskpred.mean()
    maskpred = torch.nn.functional.interpolate(maskpred, size=(h,w), mode='bicubic', align_corners=False)
    masknorm = maskpred.clone()    
    masknorm[maskpred>=threshold]=tensorone
    masknorm[maskpred<threshold]=tensorzero    
    if args.save_type in ['splash','both']:
        masknorm3=masknorm.repeat(1,3,1,1)
        if args.dim:
            imgmasked[masknorm3<threshold]/=3
        else:
            imgmasked[masknorm3<threshold]=tensorzero
        save_path=args.pred_folder+filename[:-4]
        outimage = imgmasked[0].cpu().detach().numpy()
        outimage = np.moveaxis(outimage,0,-1)#*255
        cv2.imwrite(save_path+'_pred_masked.jpg', outimage)
    if args.save_type in ['mask','both']:                
        save_path=args.pred_folder+filename[:-4]
        
        # print(masknorm.shape)
        save_image(masknorm[0], save_path +'.jpg')
    return start,stop,setuptime
    

if __name__ == '__main__':

    args = parse_args()
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: CUDA device is available. You might want to run the program with --cuda=True")
    if args.model_size not in ['small','medium','large']:
        print("WARNING. Model size of <%s> is not a valid unit. Accepted units are: small, medium, large. Defaulting to medium."%(args.model_size))
        args.model_size = 'medium'
    # network initialization
    print('Initializing model...')
    net = NetworkModule(fixed_feature_weights=False,size=args.model_size)
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
    if args.cuda:
        tensorone = torch.Tensor([1.]).cuda()
        tensorzero = torch.Tensor([0.]).cuda()
    else:
        tensorone = torch.Tensor([1.])
        tensorzero = torch.Tensor([0.])
    print('evaluating...')
    with torch.no_grad():

        ### Evaluate only one image
        if args.input.endswith('.png') or args.input.endswith('.jpg'):
            directory, filename = os.path.split(args.input)
            if not os.path.exists(args.input):
                print("The file: "+args.input+" does not exists.")
                exit()
            start,stop,_ = process_image(directory, filename,1)
            print('Predicting the image took %f seconds (with setup time)'% (stop-start))
            
        ### Evaluate the images in a folder
        else:
            if os.path.isfile(args.input):
                print("The specified file: "+args.input+" is not an jpg or png image, nor a folder containing jpg or png images. If you want to evaluate videos, use eval_video.py or demo_video.py.")
                exit()
            if not os.path.exists(args.input):
                print("The folder: "+args.input+" does not exists.")
                exit()
            dlist=os.listdir(args.input)
            dlist.sort()
            time_sum = 0
            counter = 0
            for filename in dlist:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    print("Predicting for: "+filename)
                    start,stop,setuptime = process_image(args.input,filename,counter)
                    if counter==0:
                        time_sum=stop-start
                        wsetuptime=setuptime
                    else:
                        time_sum+=stop-start
                        wsetuptime+=stop-start
                    counter=counter+1
                else:
                    continue
            if counter==0:
                print("The specified folder: "+args.input+" does not contain images.")
            else:
                print('Predicting %d images took %f seconds, with the average of %f ( with setup time: %f, average: %f)' % (counter,time_sum,time_sum/counter,wsetuptime,wsetuptime/counter))  
    

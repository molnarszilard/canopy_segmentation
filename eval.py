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
    parser.add_argument('--input_folder', dest='input_folder', default='./dataset/input_images/aghi/', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='saved_models/saved_model_1_9.pth', type=str, help='path to the model to use')
    parser.add_argument('--model_size', dest='model_size', default='large', type=str, help='size of the model: small, medium, large')
    parser.add_argument('--save_type', dest='save_type', default="mask", type=str, help='do you want to save the masked image, the mask, or both: splash, mask, both')

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

    print('evaluating...')
    with torch.no_grad():
        if args.input_folder.endswith('.png') or args.input_folder.endswith('.jpg'):
            if not os.path.exists(args.input_folder):
                print("The file: "+args.input_folder+" does not exists.")
                exit()
            img = cv2.imread(args.input_folder).astype(np.float32)
            img = cv2.resize(img,(640,480))
            img = np.moveaxis(img,-1,0)/255
            img = torch.from_numpy(img).float().unsqueeze(0)
            if args.cuda:
                img = img.cuda()
            start = timeit.default_timer()
            maskpred = net(img)
            stop = timeit.default_timer()
            threshold = maskpred.mean()
            if args.cuda:
                tensorone = torch.Tensor([1.]).cuda()
                tensorzero = torch.Tensor([0.]).cuda()
            else:
                tensorone = torch.Tensor([1.])
                tensorzero = torch.Tensor([0.])
            masknorm = maskpred.clone()
            masknorm[maskpred>=threshold]=tensorone
            masknorm[maskpred<threshold]=tensorzero
            dirname, basename = os.path.split(args.input_folder)
            if args.save_type in ['splash','both']:
                imgmasked = img.clone()
                masknorm3=masknorm.repeat(1,3,1,1)
                imgmasked[masknorm3<threshold]/=3
                save_path=args.pred_folder+basename[:-4]
                outimage = imgmasked[0].cpu().detach().numpy()
                outimage = np.moveaxis(outimage,0,-1)*255
                cv2.imwrite(save_path+'_pred_masked.jpg', outimage)
            if args.save_type in ['mask','both']:                
                save_path=args.pred_folder+basename[:-4]
                save_image(masknorm[0], save_path +'_pred.jpg')
            print('Predicting the image took %f seconds (with setup time)'% (stop-start))
        else:
            if os.path.isfile(args.input):
                print("The specified file: "+args.input+" is not an jpg or png image, nor a folder containing jpg or png images. If you want to evaluate videos, use eval_video.py or demo_video.py.")
                exit()
            if not os.path.exists(args.input_folder):
                print("The folder: "+args.input_folder+" does not exists.")
                exit()
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
                    if args.cuda:
                        img = img.cuda()
                    if counter==0:
                        start = timeit.default_timer()
                        maskpred = net(img) #in order to remove the setup-time
                        stop = timeit.default_timer()
                        setuptime = stop-start
                    start = timeit.default_timer()
                    maskpred = net(img)
                    stop = timeit.default_timer()
                    if counter==0:
                        time_sum=stop-start
                        wsetuptime=setuptime
                    else:
                        time_sum+=stop-start
                        wsetuptime+=stop-start
                    counter=counter+1
                    threshold = maskpred.mean()
                    if args.cuda:
                        tensorone = torch.Tensor([1.]).cuda()
                        tensorzero = torch.Tensor([0.]).cuda()
                    else:
                        tensorone = torch.Tensor([1.])
                        tensorzero = torch.Tensor([0.])
                    masknorm = maskpred.clone()
                    masknorm[maskpred>=threshold]=tensorone
                    masknorm[maskpred<threshold]=tensorzero
                    save_path=args.pred_folder+filename[:-4]
                    if args.save_type in ['splash','both']:
                        imgmasked = img.clone()
                        masknorm3=masknorm.repeat(1,3,1,1)
                        imgmasked[masknorm3<threshold]/=3
                        outimage = imgmasked[0].cpu().detach().numpy()
                        outimage = np.moveaxis(outimage,0,-1)*255
                        cv2.imwrite(save_path+'_pred_masked.jpg', outimage)
                    if args.save_type in ['mask','both']:         
                        save_image(masknorm[0], save_path +'_pred.jpg')
                else:
                    continue
            print('Predicting %d images took %f seconds, with the average of %f ( with setup time: %f, average: %f)' % (counter,time_sum,time_sum/counter,wsetuptime,wsetuptime/counter))  
            if counter<1:
                print("The specified folder: "+args.input_folder+" does not contain images.")
    

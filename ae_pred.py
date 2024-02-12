import argparse
import cv2
import numpy as np
import os, sys
import timeit
import torch
from network import NetworkModule
from imgaug import augmenters as iaa
import math

folder_reco = "pred_reco/"

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='canopy segmentation on video, creating another video')
    parser.add_argument('--cuda', default=True, action='store_true', help='whether use CUDA')
    parser.add_argument('--input', default='./dataset/input_images/flights/DJI_0607.mp4', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', default=None, type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', default='saved_models/saved_model_1_9.pth', type=str, help='path to the model to use')
    parser.add_argument('--model_size', default='large', type=str, help='size of the model: small, medium, large')
    parser.add_argument('--one_vid', default=True, type=bool, help='if you are processing multiple videos from a folder, do you want to create separate or only one video?')
    parser.add_argument('--frames', default=1, type=int, help='process every Xth frame from the video')
    parser.add_argument('--dim', default=False, type=bool, help='dim the pixels that are not segmented, or leave them black?')
    parser.add_argument('--cs', default='rgb', type=str, help='color space: rgb, lab')
    parser.add_argument('--save_type', default="binary", type=str, help='do you want to save the mask, the masked image, the dimmed image, or draw a contour: masking, binary, dim, contour, all')
    parser.add_argument("--proc_height", default=360,type=int,help="processing height")
    parser.add_argument("--proc_width", default=640,type=int,help="processing width")
    parser.add_argument('--start_height', default=360, type=int, help='resize the input into this size, then it will be splitted to proc_height')
    parser.add_argument('--start_width', default=640, type=int, help='resize the input into this size, then it will be splitted to proc_width')
    parser.add_argument('--confidence', default=0.5, type=float, help='confidence threshold')
    args = parser.parse_args()
    return args

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s --- %s/%s %s\r' % (bar, percents, '%', str(total), str(count), suffix))
    sys.stdout.flush()

def process_image(img_orig,initital_run):
    img_orig=cv2.resize(img_orig,(args.start_width,args.start_height))
    img_orig = np.moveaxis(img_orig,-1,0)
    img_orig = torch.from_numpy(img_orig).float().unsqueeze(0)
    if args.cuda:
        img_orig = img_orig.cuda()
    pred = net(img_orig/255)
    pred=pred[0].cpu().detach().numpy()*255
    pred = np.moveaxis(pred,0,-1)
    return pred
    

if __name__ == '__main__':

    args = parse_args()
    if args.pred_folder is None:
        args.pred_folder = args.input
    if not args.pred_folder.endswith("/"):
        args.pred_folder=args.pred_folder+"/"
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving the preds is created!")
    if not os.path.exists(args.pred_folder+folder_reco):
        os.makedirs(args.pred_folder+folder_reco)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: CUDA device is available. You might want to run the program with --cuda=True")
    if args.model_size not in ['small','medium','large']:
        print("WARNING. Model size of <%s> is not a valid unit. Accepted units are: small, medium, large. Defaulting to medium."%(args.model_size))
        args.model_size = 'medium'
    model_size = args.model_size
    model_dir, model_name = os.path.split(args.model_path)
    if "small" in model_name:
        model_size = "small"
    elif "medium" in model_name:
        model_size = "medium"
    elif "large" in model_name:
        model_size = "large"
    # network initialization
    print('Initializing model...')
    net = NetworkModule(fixed_feature_weights=False,size=model_size)
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
    print("loaded checkpoint")
    del checkpoint
    torch.cuda.empty_cache()
    net.eval()
    print('evaluating...')
    with torch.no_grad():

        ### Segment only one image
        if args.input.endswith('.png') or args.input.endswith('.jpg'):
            directory, filename = os.path.split(args.input)
            if directory[-1]!="/":
                directory=directory+"/"
            if not os.path.exists(args.input):
                print("The file: "+args.input+" does not exists.")
                exit()
            img = cv2.imread(directory+filename)
            pred = process_image(img,0)
            cv2.imwrite(args.pred_folder+folder_reco+filename, pred)


        ### Segment the frames of a video, and save them separately
        elif args.input.endswith('.mp4'):
            directory, filename = os.path.split(args.input)
            if directory[-1]!="/":
                directory=directory+"/"
            if not os.path.exists(args.input):
                print("The file: "+args.input+" does not exists.")
                exit()
            cap = cv2.VideoCapture(args.input)
            totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("total frames in video: "+str(totalFrames))
            time_sum_imgs=0
            time_sum_splits=0
            print("\n")
            for currentFrame in range(0,int(totalFrames),args.frames):
                progress(currentFrame,totalFrames,"frames")
                cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame)
                ret, img = cap.read()
                pred = process_image(img,0)
                
                number=f'{currentFrame:05d}'
                cv2.imwrite(args.pred_folder+folder_reco+filename[:-4]+"_f_"+str(number)+".png", pred)
                
            cap.release()
            print("\n")


        ### Segment the images in a folder
        else:
            if os.path.isfile(args.input):
                print("The specified file: "+args.input+" is not an jpg or png image, nor a folder containing jpg or png images. If you want to evaluate videos, use eval_video.py or demo_video.py.")
                exit()
            if not os.path.exists(args.input):
                print("The folder: "+args.input+" does not exists.")
                exit()
            dlist=os.listdir(args.input)
            dlist.sort()
            time_sum_imgs = 0
            time_sum_splits= 0
            counter = 0
            print("\n")
            for filename in dlist:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    progress(counter,len(dlist),filename)
                    # print("Predicting for: "+filename)
                    img = cv2.imread(args.input+filename)
                    pred = process_image(img,0)
                    cv2.imwrite(args.pred_folder+folder_reco+filename, pred)
                    counter+=1
                else:
                    continue
            print("\n")
            if counter==0:
                print("The specified folder: "+args.input+" does not contain images.")    

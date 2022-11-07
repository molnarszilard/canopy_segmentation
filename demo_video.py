import argparse
from pickletools import uint8
import cv2
import numpy as np
import os
import timeit
import torch
from torchvision.utils import save_image
from network import NetworkModule
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='canopy segmentation on video, creating another video')
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true', help='whether use CUDA')
    parser.add_argument('--input', dest='input', default='./dataset/input_images/flights/DJI_0607.mp4', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='saved_models/saved_model_1_9.pth', type=str, help='path to the model to use')
    parser.add_argument('--one_vid', dest='one_vid', default=True, type=bool, help='if you are processing multiple videos from a folder, do you want to create separate or only one video?')
    parser.add_argument('--frames', dest='frames', default=1, type=int, help='process every Xth frame from the video')

    args = parser.parse_args()
    return args

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s --- %s/%s %s\r' % (bar, percents, '%', str(total), str(count), suffix))
    sys.stdout.flush()

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
    img_array = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    myFrameNumber = args.frames
    with torch.no_grad():
        if args.input.endswith('.mp4'):
                dirname, basename = os.path.split(args.input)
                save_path=args.pred_folder+basename[:-4]
                print("processing: "+args.input)
                cap = cv2.VideoCapture(args.input)
                totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print("total frames in video: "+str(totalFrames))
                fps = cap.get(cv2.CAP_PROP_FPS)
                video = cv2.VideoWriter(save_path+"_segmented.mp4", fourcc, fps, (640,480))
                currentFrame = 0
                while currentFrame<totalFrames:
                    progress(currentFrame,totalFrames,"frames")
                    cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame)
                    ret, img = cap.read()
                    try:
                            img = cv2.resize(img,(640,480))
                    except:
                        print("Something went wrong processing resize.")
                        break
                    img = np.moveaxis(img,-1,0)/255
                    img = torch.from_numpy(img).float().unsqueeze(0)
                    img = img.cuda()
                    maskpred = net(img)
                    threshold = maskpred.mean()
                    imgmasked = img.clone()
                    imgmasked[maskpred<=threshold]/=3 
                    outimage = imgmasked[0].cpu().detach().numpy()
                    outimage = np.moveaxis(outimage,0,-1)*255
                    video.write(outimage.astype(np.uint8))
                    currentFrame = currentFrame + myFrameNumber
                cap.release()
                video.release()
                print("done")
        else:
            dlist=os.listdir(args.input)
            dlist.sort()
            fps = 0
            video = None
            for filename in dlist:
                if filename.endswith(".mp4"):
                    print("processing: "+args.input+filename)
                    cap = cv2.VideoCapture(args.input+filename)
                    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    print("total frames in video: "+str(totalFrames))
                    if args.one_vid:
                        save_path=args.pred_folder+"full_segmented_video.mp4"
                    else:
                        save_path=args.pred_folder+filename[:-4]+"_segmented.mp4"
                    if video == None or not args.one_vid:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        video = cv2.VideoWriter(save_path, fourcc, int(fps), (640,480))
                    currentFrame = 0
                    while currentFrame<totalFrames:
                        progress(currentFrame,totalFrames,"frames")
                        cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame)
                        ret, img = cap.read()
                        try:
                            img = cv2.resize(img,(640,480))
                        except:
                            print("Something went wrong processing resize.")
                            break
                        img = np.moveaxis(img,-1,0)/255
                        img = torch.from_numpy(img).float().unsqueeze(0)
                        img = img.cuda()
                        maskpred = net(img)
                        threshold = maskpred.mean()
                        # tensorzero = torch.Tensor([0.]).cuda()
                        # tensorone = torch.Tensor([1.]).cuda()
                        imgmasked = img.clone()
                        maskpred3=maskpred.repeat(1,3,1,1)
                        imgmasked[maskpred3<=threshold]/=3
                        outimage = imgmasked[0].cpu().detach().numpy()
                        outimage = np.moveaxis(outimage,0,-1)*255
                        video.write(outimage.astype(np.uint8))
                        currentFrame = currentFrame + myFrameNumber
                    cap.release()
                    if not args.one_vid:
                        video.release()
                    print("done")
            if args.one_vid and video != None:
                video.release()

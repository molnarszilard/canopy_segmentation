import argparse
import cv2
import numpy as np
import os
import torch
from network import NetworkModule
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='canopy segmentation, individual frames from video')
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true', help='whether use CUDA')
    parser.add_argument('--input', dest='input', default='./dataset/input_images/flights/DJI_0607.mp4', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='saved_models/saved_model_1_9.pth', type=str, help='path to the model to use')

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
    with torch.no_grad():
        if args.input.endswith('.mp4'):            
                myFrameNumber = 30
                print("processing: "+args.input)
                cap = cv2.VideoCapture(args.input)
                totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print("total frames in video: "+str(totalFrames))
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
                    if args.cuda:
                        img = img.cuda()
                    maskpred = net(img)
                    threshold = maskpred.mean()
                    imgmasked = img.clone()
                    maskpred3=maskpred.repeat(1,3,1,1)
                    imgmasked[maskpred3<=threshold]/=3
                    dirname, basename = os.path.split(args.input)
                    save_path=args.pred_folder+basename[:-4]
                    number=f'{currentFrame:05d}'
                    outimage = imgmasked[0].cpu().detach().numpy()
                    outimage = np.moveaxis(outimage,0,-1)*255
                    cv2.imwrite(save_path+"_f_"+number+"_pred"+'.png', outimage)
                    currentFrame = currentFrame + myFrameNumber
                cap.release()
                print("done")
        else:
            dlist=os.listdir(args.input)
            dlist.sort()
            for filename in dlist:
                if filename.endswith(".mp4"):
                    myFrameNumber = 30
                    print("processing: "+args.input+filename)
                    cap = cv2.VideoCapture(args.input+filename)
                    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    print("total frames in video: "+str(totalFrames))
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
                        if args.cuda:
                            img = img.cuda()
                        maskpred = net(img)
                        threshold = maskpred.mean()
                        imgmasked = img.clone()
                        maskpred3=maskpred.repeat(1,3,1,1)
                        imgmasked[maskpred3<=threshold]/=3
                        save_path=args.pred_folder+filename[:-4]
                        number=f'{currentFrame:05d}'
                        outimage = imgmasked[0].cpu().detach().numpy()
                        outimage = np.moveaxis(outimage,0,-1)*255
                        cv2.imwrite(save_path+"_f_"+number+"_pred"+'.png', outimage)
                        currentFrame = currentFrame + myFrameNumber
                    cap.release()
                    print("done")

import argparse
import cv2
import numpy as np
import os, sys
import timeit
import torch
from network import NetworkModule
from imgaug import augmenters as iaa
import math

folder_binary = "pred_masks/"
folder_masking = "pred_masked/"
folder_dimmed = "pred_dimmed/"
folder_contoured = "pred_contoured/"

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
    parser.add_argument('--confidence', default=0.5, type=float, help='confidence threshold, if confidence equals zero or None, not thresholding is performed')
    parser.add_argument('--contourwidth', default=1, type=int, help='width of the contour line')
    args = parser.parse_args()
    return args

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s --- %s/%s %s\r' % (bar, percents, '%', str(total), str(count), suffix))
    sys.stdout.flush()

def split(img):
    splitted_images = []
    h,w,_ = img.shape
    if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 512:
        print("Image does not contain enough information (mostly black). Skipping...")
        exit()
    hn=1 ##normalized height
    wn=1 ##normalized width
    thn=float(args.proc_height/h) ##normalized target height
    twn=float(args.proc_width/w) ##normalized target width
    crop_on_height = True
    crop_on_width = True
    if thn>=1.0:
        crop_on_height=False
    else:
        crop_h = int(hn/thn) ##number of primary crops on height    
        crop_h_mid = crop_h-1 ##number of secondary crops on height
        if crop_h==1 and thn<1:
            crop_h_mid=1
        rem_h = hn-float(crop_h*thn) ##remaining part on the height    
        dis_h = float(rem_h/crop_h_mid) ## distance between crops on the heigth    
        step_h = (thn+dis_h)/2 ## step to increase the start of crops on the height  
        if crop_h==1 and thn<1:
            step_h=dis_h     
    if twn>=1.0:
        crop_on_width=False
    else:
        crop_w = int(wn/twn) ##number of primary crops on width
        crop_w_mid = crop_w-1 ##number of secondary crops on width
        if crop_w==1 and twn<1:
            crop_w_mid=1
        rem_w = wn-float(crop_w*twn) ##how many crops do you need on the width
        dis_w = float(rem_w/crop_w_mid) ## distance between crops on the width
        step_w = (twn+dis_w)/2 ## step to increase the start of crops on the width  
        if crop_w==1 and twn<1: 
            step_w=dis_w
    if crop_on_height and crop_on_width:
        posh=0
        for i in range(crop_h+crop_h_mid):            
            posw=0
            for j in range(crop_w+crop_w_mid):
                # print("%d, %d, %d, %d"%(posh,thn,posw,twn))
                top=posh
                bottom=1.0-thn-posh
                right=1.0-twn-posw
                left=posw
                if bottom<0 and bottom>-1/h:
                    bottom=abs(bottom)
                if right<0 and right>-1/w:
                    right=abs(right)
                if top<0 or top>1+1/h or bottom<0 or bottom>1+1/h or left<0 or left>1+1/w or right<0 or right>1+1/w:
                    print("%f, %f, %f, %f"%(top,bottom,left,right))
                    print("%f, %f, %f, %f"%(posh,thn,posw,twn))
                    continue
                cropping = iaa.Crop(percent=(top, right, bottom, left), keep_size=False) #top, right, bottom, left
                imgmod = cropping(image=img)
                splitted_images.append(imgmod)
                posw=posw+step_w
            posh=posh+step_h
    elif crop_on_height:
        posh=0
        for i in range(crop_h+crop_h_mid):
            top=posh
            bottom=1.0-thn-posh
            right=0.0
            left=0.0
            if bottom<0 and bottom>-1/h:
                bottom=abs(bottom)
            if top<0 or top>1+1/h or bottom<0 or bottom>1+1/h:
                continue
            cropping = iaa.Crop(percent=(top, right, bottom, left), keep_size=False) #top, right, bottom, left
            imgmod = cropping(image=img)
            splitted_images.append(imgmod)
            posh=posh+step_h
    elif crop_on_width:
        posw=0
        for i in range(crop_w+crop_w_mid):
            top=0.0
            bottom=0.0
            right=1.0-twn-posw+0
            left=posw
            if right<0 and right>-1/w:
                right=abs(right)
            if left<0 or left>1+1/w or right<0 or right>1+1/w:
                continue
            cropping = iaa.Crop(percent=(top, right, bottom, left), keep_size=False) #top, right, bottom, left
            imgmod = cropping(image=img)
            splitted_images.append(imgmod)
            posw=posw+step_w
    else:
        splitted_images.append(imgmod)
    return splitted_images

def assemble(img,masks):
    h,w,_ = img.shape
    if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 512:
        print("Image does not contain enough information (mostly black). Skipping...")
        exit()
    maskpred = np.zeros((h,w), np.uint8)
    hn=1 ##normalized height
    wn=1 ##normalized width
    thn=float(args.proc_height/h) ##normalized target height
    twn=float(args.proc_width/w) ##normalized target width
    crop_on_height = True
    crop_on_width = True
    if thn>=1.0:
        crop_on_height=False
    else:
        crop_h = int(hn/thn) ##number of primary crops on height    
        crop_h_mid = crop_h-1 ##number of secondary crops on height
        if crop_h==1 and thn<1:
            crop_h_mid=1
        rem_h = hn-float(crop_h*thn) ##remaining part on the height    
        dis_h = float(rem_h/crop_h_mid) ## distance between crops on the heigth    
        step_h = (thn+dis_h)/2 ## step to increase the start of crops on the height  
        if crop_h==1 and thn<1:
            step_h=dis_h     
    if twn>=1.0:
        crop_on_width=False
    else:
        crop_w = int(wn/twn) ##number of primary crops on width
        crop_w_mid = crop_w-1 ##number of secondary crops on width
        if crop_w==1 and twn<1:
            crop_w_mid=1
        rem_w = wn-float(crop_w*twn) ##how many crops do you need on the width
        dis_w = float(rem_w/crop_w_mid) ## distance between crops on the width
        step_w = (twn+dis_w)/2 ## step to increase the start of crops on the width  
        if crop_w==1 and twn<1: 
            step_w=dis_w
    index=0
    weights = np.zeros_like(masks[index],np.float32)
    maxdist = math.sqrt((weights.shape[1])**2+(weights.shape[0])**2)/2
    for i in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            dist = math.sqrt((weights.shape[1]/2-i)**2+(weights.shape[0]/2-j)**2)
            weights[j,i]=1-dist/maxdist/1.5
    if crop_on_height and crop_on_width:
        posh=0
        for i in range(crop_h+crop_h_mid):
            posw=0
            for j in range(crop_w+crop_w_mid):
                top=int(posh*h)
                bottom=top+masks[index].shape[0]
                left=int(posw*w)
                right=left+masks[index].shape[1]
                if top<0 or top>h or bottom<0 or bottom>h or left<0 or left>w or right<0 or right>w:
                    continue
                np.add(maskpred[top:bottom, left:right], masks[index], out=maskpred[top:bottom, left:right], casting="unsafe")
                index=index+1
                posw=posw+step_w
            posh=posh+step_h
    elif crop_on_height:
        posh=0
        for i in range(crop_h+crop_h_mid):
            top=int(posh*h)
            bottom=top+masks[index].shape[0]
            left=int(0)
            right=masks[index].shape[1]
            if top<0 or top>h or bottom<0 or bottom>h:
                continue
            # masks[index]=np.multiply(masks[index],weights)
            np.add(maskpred[top:bottom, left:right], masks[index], out=maskpred[top:bottom, left:right], casting="unsafe")
            # maskpred[top:bottom, left:right] += masks[index]
            index=index+1
            posh=posh+step_h
    elif crop_on_width:
        posw=0
        for i in range(crop_w+crop_w_mid):
            top=0
            bottom=masks[index].shape[0]
            left=int(posw*w)
            right=left+masks[index].shape[1]
            if left<0 or left>w or right<0 or right>w:
                continue
            # masks[index]=np.multiply(masks[index],weights)
            np.add(maskpred[top:bottom, left:right], masks[index], out=maskpred[top:bottom, left:right], casting="unsafe")
            # maskpred[top:bottom, left:right] += masks[index]
            index=index+1
            posw=posw+step_w
    else:
        maskpred = masks[0]
        index=index+1
    # maskpred=maskpred/int(maskpred.max())*255
    if args.confidence>0:
        maskpred = np.where(maskpred<args.confidence*255,0,255)
    # print(maskpred.max())
    return maskpred

def process_image(img_orig,initital_run):
    # img_orig = cv2.imread(directory+filename)
    start_img = timeit.default_timer()
    imgmasked = img_orig.copy()
    if args.start_width is not None and args.start_height is not None:
        img_orig=cv2.resize(img_orig,(args.start_width,args.start_height))
    proc_img = img_orig.copy()
    # print("Initial size of image: %dx%d"%(h,w))
    if proc_img.shape[0]>args.proc_height or proc_img.shape[1]>args.proc_width:
        splitted_images=split(proc_img)
    else:
        splitted_images = []
        splitted_images.append(cv2.resize(proc_img,(args.proc_width,args.proc_height)))
    result_masks = []
    duration_split = 0
    init_duration = 0
    for i in range(len(splitted_images)):
        sub_image = splitted_images[i].copy()
        if sub_image.shape[0]!=args.proc_height or sub_image.shape[1]!=args.proc_width:
            print(splitted_images[i].shape)
            print(sub_image.shape)
            sub_image = cv2.resize(sub_image,(args.proc_width,args.proc_height))
        sub_image = np.moveaxis(sub_image,-1,0)
        sub_image = torch.from_numpy(sub_image).float().unsqueeze(0)
        if args.cuda:
            sub_image = sub_image.cuda()
        if not initital_run and i==0:
            start = timeit.default_timer()
            _ = net(sub_image/255) #in order to remove the setup-time
            stop = timeit.default_timer()
            init_duration=stop-start
        start = timeit.default_timer()
        sub_maskpred = net(sub_image/255)
        stop = timeit.default_timer()
        duration_split+=stop-start
        sub_maskpred=sub_maskpred[0,0].cpu().detach().numpy()*255       
        # sub_maskpred = np.moveaxis(sub_maskpred,0,-1)
        if args.confidence>0:
            sub_maskpred = np.where(sub_maskpred<args.confidence*255,0,255)
        if splitted_images[i].shape[0]!=args.proc_height or splitted_images[i].shape[1]!=args.proc_width:
            sub_maskpred = cv2.resize(sub_maskpred,(splitted_images[i].shape[1],splitted_images[i].shape[0]))
            if args.confidence>0:
                sub_maskpred = np.where(sub_maskpred<args.confidence*255,0,255)
        result_masks.append(sub_maskpred)

    if len(result_masks)>1:
        maskpred=assemble(proc_img,result_masks).astype(np.float32)
    else:
        maskpred=result_masks[0]
    if args.confidence>0:
            maskpred = np.where(maskpred<args.confidence*255,0,255).astype(np.float32)
    maskpred = cv2.resize(maskpred,(imgmasked.shape[1],imgmasked.shape[0])).astype(np.uint8)
    if args.confidence>0:
        maskpred = np.where(maskpred<args.confidence*255,0,255)
    masknorm3 = np.zeros((maskpred.shape[0],maskpred.shape[1],3), np.uint8)
    masknorm3[:,:,0]=maskpred
    masknorm3[:,:,1]=maskpred
    masknorm3[:,:,2]=maskpred
    ret_mask = None
    ret_masked = None
    ret_dimmed = None
    ret_contoured = None
    if "binary" in args.save_type or args.save_type=="all":
        ret_mask=masknorm3
    if "masking" in args.save_type or args.save_type=="all":
        imgmod = imgmasked.copy()
        imgmod = np.where(masknorm3<args.confidence*255,0.0,imgmod)
        ret_masked=imgmod
    if "dim" in args.save_type or args.save_type=="all":
        _, thresh = cv2.threshold(maskpred.astype(np.uint8) , args.confidence*255, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgmod2 = imgmasked.copy()
        imgmod2 = np.where(masknorm3<args.confidence*255,imgmod2/2,imgmod2)
        cv2.drawContours(imgmod2, contours, -1, (0, 0, 255), 1)
        ret_dimmed=imgmod2
    if "contour" in args.save_type or args.save_type=="all":
        _, thresh = cv2.threshold(maskpred.astype(np.uint8) , args.confidence*255, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgmod3 = imgmasked.copy()
        cv2.drawContours(imgmod3, contours, -1, (0, 0, 255), args.contourwidth)
        ret_contoured=imgmod3
    stop_img = timeit.default_timer()
    duration_img = stop_img-start_img-init_duration
    return ret_mask, ret_masked, ret_dimmed, ret_contoured, duration_img, duration_split
    

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
    if ("binary" in args.save_type or args.save_type=="all") and not os.path.exists(args.pred_folder+folder_binary):
        os.makedirs(args.pred_folder+folder_binary)
    if ("masking" in args.save_type or args.save_type=="all") and not os.path.exists(args.pred_folder+folder_masking):
        os.makedirs(args.pred_folder+folder_masking)
    if ("dim" in args.save_type or args.save_type=="all") and not os.path.exists(args.pred_folder+folder_dimmed):
        os.makedirs(args.pred_folder+folder_dimmed)
    if ("contour" in args.save_type or args.save_type=="all") and not os.path.exists(args.pred_folder+folder_contoured):
        os.makedirs(args.pred_folder+folder_contoured)
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
            res_binary, res_masked, res_dimmed, res_contoured, duration_img, duration_splits = process_image(img,0)
            if "binary" in args.save_type or args.save_type=="all":
                cv2.imwrite(args.pred_folder+folder_binary+filename, res_binary)
            if "masking" in args.save_type or args.save_type=="all":
                cv2.imwrite(args.pred_folder+folder_masking+filename, res_masked)
            if "dim" in args.save_type or args.save_type=="all":
                cv2.imwrite(args.pred_folder+folder_dimmed+filename, res_dimmed)
            if "contour" in args.save_type or args.save_type=="all":
                cv2.imwrite(args.pred_folder+folder_contoured+filename, res_contoured)
            print('Segmenting the image took a total of %f seconds (%f seconds in network)'%(duration_img,duration_splits))


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
                res_binary, res_masked, res_dimmed, res_contoured, duration_img, duration_splits = process_image(img,0)
                time_sum_imgs+=duration_img
                time_sum_splits+=duration_splits
                number=f'{currentFrame:05d}'
                if "binary" in args.save_type or args.save_type=="all":
                    cv2.imwrite(args.pred_folder+folder_binary+filename[:-4]+"_f_"+str(number)+".png", res_binary)
                if "masking" in args.save_type or args.save_type=="all":
                    cv2.imwrite(args.pred_folder+folder_masking+filename[:-4]+"_f_"+str(number)+".png", res_masked)
                if "dim" in args.save_type or args.save_type=="all":
                    cv2.imwrite(args.pred_folder+folder_dimmed+filename[:-4]+"_f_"+str(number)+".png", res_dimmed)
                if "contour" in args.save_type or args.save_type=="all":
                    cv2.imwrite(args.pred_folder+folder_contoured+filename[:-4]+"_f_"+str(number)+".png", res_contoured)
            cap.release()
            print("\n")
            print('Segmenting the video took a total of %f seconds (%f seconds in network)'%(time_sum_imgs,time_sum_splits))


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
                    res_binary, res_masked, res_dimmed, res_contoured, duration_img, duration_splits = process_image(img,counter)
                    if "binary" in args.save_type or args.save_type=="all":
                        cv2.imwrite(args.pred_folder+folder_binary+filename, res_binary)
                    if "masking" in args.save_type or args.save_type=="all":
                        cv2.imwrite(args.pred_folder+folder_masking+filename, res_masked)
                    if "dim" in args.save_type or args.save_type=="all":
                        cv2.imwrite(args.pred_folder+folder_dimmed+filename, res_dimmed)
                    if "contour" in args.save_type or args.save_type=="all":
                        cv2.imwrite(args.pred_folder+folder_contoured+filename, res_contoured)
                    time_sum_imgs+=duration_img
                    time_sum_splits+=duration_splits
                    counter+=1
                else:
                    continue
            print("\n")
            if counter==0:
                print("The specified folder: "+args.input+" does not contain images.")
            else:
                print('Segmenting %d images took a total of %f seconds, with the average of %f ( in network: %f, average: %f)' % (counter,time_sum_imgs,time_sum_imgs/counter,time_sum_splits,time_sum_splits/counter))  
    

import os
import torch
import time
import argparse
from dataloader import DataLoader
from network import NetworkModule
import numpy as np
import sys
from torchsummary import summary
import cv2
from metrics import RMSELoss
from metrics import LossIoU

def progress(count, total, epoch, suffix):
    bar_len = 10
    filled_len = int(bar_len * count / float(total))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('Epoch: %s [%s] %s%s --- %s/%s %s\r' % (epoch,bar, percents, '%', str(total), str(count), suffix))
    sys.stdout.flush()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Segmenting vine canopy')
    parser.add_argument('--bs', dest='bs', default=4, type=int, help='batch_size')
    parser.add_argument('--checkepoch', dest='checkepoch', default=None, type=str, help='checkepoch to load model')
    parser.add_argument('--cuda', dest='cuda', default=True, help='whether use CUDA')
    parser.add_argument('--data_dir', dest='data_dir', default='./dataset/canopy_mask_dataset/group1/', type=str, help='dataset directory')
    parser.add_argument('--dir_images', dest='dir_images', default='training_images/', type=str, help='directory where to save the training images')
    parser.add_argument('--epochs', dest='max_epochs', default=10, type=int, help='number of epochs to train')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', default=4e-5, type=float, help='learning rate decay ratio')
    parser.add_argument('--lr', dest='lr', default=1e-5, type=float, help='starting learning rate')
    parser.add_argument('--model_dir', dest='model_dir', default='saved_models', type=str, help='output directory')
    parser.add_argument('--model_size', dest='model_size', default='large', type=str, help='size of the model: small, medium, large')
    parser.add_argument('--num_workers', dest='num_workers', default=8, type=int, help='num_workers')
    parser.add_argument('--o', dest='optimizer', default="adam", type=str, help='training optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='training momentum')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for adam optimizer')
    parser.add_argument('--r', dest='resume', default=False, type=bool, help='resume checkpoint or not')
    parser.add_argument('--s', dest='session', default=1, type=int, help='training session')
    parser.add_argument('--save_epoch', dest='save_epoch', default=5, type=int, help='after how many epochs do you want the model to be saved')
    parser.add_argument('--save_images', dest='save_images', default=100, type=int, help='save every x-th image during the training to see its evolution, 0 - means off')
    parser.add_argument('--start_at', dest='start_epoch', default=0, type=int, help='epoch to start with')
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
    parser.add_argument('--img_height', dest='img_height', default=360, type=int, help='resize the input images to this height')
    parser.add_argument('--img_width', dest='img_width', default=640, type=int, help='resize the input images to this width')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    isExist = os.path.exists(args.model_dir)
    if not isExist:
        os.makedirs(args.model_dir)
        print("The new directory for saving models while training is created: "+args.model_dir)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: CUDA device is available. You might want to run the program with --cuda=True")

    if args.model_size not in ['small','medium','large']:
        print("WARNING. Model size of <%s> is not a valid unit. Accepted units are: small, medium, large. Defaulting to medium."%(args.model_size))
        args.model_size = 'medium'
    train_dataset = DataLoader(root=args.data_dir,train=True,cs=args.cs,img_height=args.img_height,img_width=args.img_width)
    train_size = len(train_dataset)
    eval_dataset = DataLoader(root=args.data_dir,train=False,cs=args.cs,img_height=args.img_height,img_width=args.img_width)
    eval_size = len(eval_dataset)
    print(train_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)

    # network initialization
    print('Initializing model...')
    model_size = args.model_size
    if args.checkepoch is not None:
        if "small" in args.checkepoch:
            model_size = "small"
        elif "medium" in args.checkepoch:
            model_size = "medium"
        elif "large" in args.checkepoch:
            model_size = "large"
    net = NetworkModule(fixed_feature_weights=False,size=model_size)
    if args.cuda:
        net = net.cuda()
    print("Model initialization done.")

    print("Setting up parameters...")
    lr = args.lr
    bs = args.bs
    params = []
    for key, value in dict(net.named_parameters()).items():
      if value.requires_grad:
        params += [{'params':[value],'lr':lr, 'weight_decay': args.lr_decay_gamma}]
    print("Parameters are set.")

    print("Configuring optimizer...")
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(args.momentum, args.momentum*1.11), eps=args.eps, weight_decay=args.lr_decay_gamma)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum)
    print("Optimizer configured.")

    # resume
    if args.resume:
        load_name = os.path.join(args.model_dir,args.checkepoch)
        

        print("loading checkpoint %s" % (load_name))
        state = net.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        net.load_state_dict(state)
        if model_size not in [net.get_size()]:
            print("WARNING. Model size of <%s> is not equal to the size in the saved model. The correct model size for this is: %s"%(model_size,net.get_size()))
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()

    loss_iou = LossIoU()
    loss_rmse = RMSELoss()
    # summary(net, (3, train_dataset.height, train_dataset.width),4)
    # print(net)
    iters_per_epoch = int(train_size / args.bs)
    min_eval_loss = 1000.0
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        net.train()
        start = time.time()        
        train_data_iter = iter(train_dataloader)
        for step in range(iters_per_epoch):
            data = train_data_iter.next()            
            img,maskgt=data
            if args.cuda:
                img = img.cuda()
                maskgt = maskgt.cuda()
            optimizer.zero_grad()
            maskpred = net(img/255)
            loss_train_iou = loss_iou(maskpred, maskgt)
            loss_train_rmse = loss_rmse(maskpred, maskgt)
            # print("IoU: %f, RMSE: %f"%(loss_train_iou,loss_train_rmse))
            loss = (loss_train_iou+loss_train_rmse)
            # loss = loss_train_rmse
            # loss.requires_grad =True 
            loss.backward()
            optimizer.step()
            # info
            progress(step,iters_per_epoch-1,epoch,"Iou: %f, RMSE: %f"%(loss_train_iou.item(),loss_train_rmse.item()))
        print("training epoch %d done - "%(epoch))
        end = time.time()
        # print('time elapsed: %fs' % (end - start))        
        # print('evaluating...')
        eval_loss = 0
        eval_loss_iou = 0
        eval_loss_rmse = 0
        with torch.no_grad():
            # setting to eval mode
            net.eval()
            if args.cuda:
                tensorone = torch.Tensor([1.]).cuda()
                tensorzero = torch.Tensor([0.]).cuda()
                tensorthreshold = torch.Tensor([0.5]).cuda()
            else:
                tensorone = torch.Tensor([1.])
                tensorzero = torch.Tensor([0.])
                tensorthreshold = torch.Tensor([0.5])
            eval_data_iter = iter(eval_dataloader)
            eval_iter = 0
            for i, data in enumerate(eval_data_iter):
                # print(i,'/',len(eval_data_iter)-1)
                progress(i,len(eval_data_iter)-1,epoch," iters")
                # data = eval_data_iter.next()  
                img,maskgt=data
                if args.cuda:
                    img = img.cuda()
                    maskgt = maskgt.cuda()            
                maskpred = net(img/255)
                
                loss_eval_iou = loss_iou(maskpred, maskgt)
                loss_eval_rmse = loss_rmse(maskpred, maskgt)
                eval_iter += 1
                eval_loss += (loss_eval_iou+loss_eval_rmse)
                eval_loss_iou += loss_eval_iou
                eval_loss_rmse += loss_eval_rmse
                if args.save_images and i%100==0:
                    if not os.path.exists(args.dir_images):
                        os.makedirs(args.dir_images)
                    # sample_mask = maskpred[0].cpu().detach().numpy()
                    numbere=f'{epoch:02d}'
                    numberi=f'{i:05d}'
                    filename = args.dir_images+'trainingpred_'+numbere+'_'+numberi+'.png'
                    filename = str(filename)

                    imgmasked = img.clone()
                    masknorm = maskpred.clone()    
                    masknorm[maskpred>=tensorthreshold]=tensorone
                    masknorm[maskpred<tensorthreshold]=tensorzero
                    masknorm3=masknorm.repeat(1,3,1,1)
                    imgmasked[masknorm3<tensorthreshold]/=3
                    # save_path=args.pred_folder+filename[:-4]
                    outimage = imgmasked[0].cpu().detach().numpy()
                    outimage = np.moveaxis(outimage,0,-1)
                    # save_image(outimage, filename)  
                    cv2.imwrite(filename, outimage)  
                    # cv2.imwrite(filename, maskpred[0].cpu().detach().numpy()*255)      
                
            eval_loss = eval_loss/eval_iter
            eval_loss_iou = eval_loss_iou/eval_iter
            eval_loss_rmse = eval_loss_rmse/eval_iter

            print("eval done")
            print("[epoch %2d] loss: %.4f , Iou: %.4f, RMSE: %.4f" \
                            % (epoch, eval_loss,eval_loss_iou,eval_loss_rmse))
            with open(os.path.join(args.model_dir, 'training_log_{}.txt'.format(args.session)), 'a') as f:
                f.write("[epoch %2d] accuracy: %.4f\n" \
                            % (epoch, eval_loss))
            if eval_loss<min_eval_loss:
                min_eval_loss=eval_loss
                if not os.path.exists(args.model_dir):
                            os.makedirs(args.model_dir)
                save_name = os.path.join(args.model_dir, 'canopy_model_{}_s{}_best.pth'.format(model_size,args.session, epoch))
                torch.save({'epoch': epoch+1, 'model': net.state_dict(), }, save_name)
                print('save model: {}'.format(save_name))
            if epoch%args.save_epoch==0 or epoch==args.max_epochs-1:
                if not os.path.exists(args.model_dir):
                            os.makedirs(args.model_dir)
                save_name = os.path.join(args.model_dir, 'canopy_model_{}_s{}_e{}.pth'.format(model_size,args.session, epoch))
                torch.save({'epoch': epoch+1, 'model': net.state_dict(), }, save_name)
                print('save model: {}'.format(save_name))
        
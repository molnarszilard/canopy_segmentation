import os
import torch
from torchvision.utils import save_image
import torch.nn as nn
import time
import argparse
from dataloader import DataLoader
from network import NetworkModule
import numpy as np
import sys
from torchsummary import summary

def progress(count, total, epoch, suffix=''):
    bar_len = 10
    filled_len = int(round(bar_len * count / float(total)))

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
    parser.add_argument('--checkepoch', dest='checkepoch', default=9, type=int, help='checkepoch to load model')
    parser.add_argument('--cuda', dest='cuda', default=True, help='whether use CUDA')
    parser.add_argument('--data_dir', dest='data_dir', default='./dataset/canopy_mask_dataset/group1/', type=str, help='dataset directory')
    parser.add_argument('--dir_images', dest='dir_images', default='training_images/', type=str, help='directory where to save the training images')
    parser.add_argument('--disp_interval', dest='disp_interval', default=10, type=int, help='display interval')
    parser.add_argument('--epochs', dest='max_epochs', default=10, type=int, help='number of epochs to train')
    parser.add_argument('--loss_multiplier', dest='loss_multiplier', default=1, type=int, help='increase the loss for much faster training')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', default=0.1, type=float, help='learning rate decay ratio')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', default=5, type=int, help='step to do learning rate decay, unit is epoch')
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float, help='starting learning rate')
    parser.add_argument('--model_dir', dest='model_dir', default='saved_models', type=str, help='output directory')
    parser.add_argument('--model_size', dest='model_size', default='medium', type=str, help='size of the model: small, medium, large')
    parser.add_argument('--num_workers', dest='num_workers', default=1, type=int, help='num_workers')
    parser.add_argument('--o', dest='optimizer', default="adam", type=str, help='training optimizer')
    parser.add_argument('--r', dest='resume', default=False, type=bool, help='resume checkpoint or not')
    parser.add_argument('--s', dest='session', default=1, type=int, help='training session')
    parser.add_argument('--save_epoch', dest='save_epoch', default=5, type=int, help='after how many epochs do you want the model to be saved')
    parser.add_argument('--save_images', dest='save_images', default=100, type=int, help='save every x-th image during the training to see its evolution, 0 - means off')
    parser.add_argument('--start_at', dest='start_epoch', default=0, type=int, help='epoch to start with')
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
    
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

class LossIoU(nn.Module):
    def __init__(self):
        super(LossIoU, self).__init__()

    def forward(self, pred, gt):
        intersection_tensor=pred*gt
        intersection = torch.sum(intersection_tensor, dim = (0,1,2,3))
        union_tensor = pred+gt-intersection_tensor
        union = torch.sum(union_tensor, dim = (0,1,2,3))
        loss = intersection/union
        return 1-loss

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
    train_dataset = DataLoader(root=args.data_dir,train=True,cs=args.cs)
    train_size = len(train_dataset)
    eval_dataset = DataLoader(root=args.data_dir,train=False,cs=args.cs)
    eval_size = len(eval_dataset)
    print(train_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)

    # network initialization
    print('Initializing model...')
    net = NetworkModule(fixed_feature_weights=False,size=args.model_size)
    if args.cuda:
        net = net.cuda()
    print("Model initialization done.")

    print("Setting up parameters...")
    lr = args.lr
    bs = args.bs
    params = []
    for key, value in dict(net.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
            DOUBLE_BIAS=0
            WEIGHT_DECAY=4e-5
            params += [{'params':[value],'lr':lr*(DOUBLE_BIAS + 1), \
                  'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': 4e-5}]
    print("Parameters are set.")

    print("Configuring optimizer...")
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    print("Optimizer configured.")

    # resume
    if args.resume:
        load_name = os.path.join(args.model_dir,
          'saved_model_{}_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        state = net.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        net.load_state_dict(state)
        if args.model_size not in [net.get_size()]:
            print("WARNING. Model size of <%s> is not equal to the size in the saved model. The correct model size for this is: %s"%(args.model_size,net.get_size()))
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()

    loss_iou = LossIoU()
    summary(net, (3, 480, 640),4)
    # print(net)
    iters_per_epoch = int(train_size / args.bs)
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        net.train()
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        
        train_data_iter = iter(train_dataloader)
        for step in range(iters_per_epoch):
            data = train_data_iter.next()            
            img,maskgt=data
            if args.cuda:
                img = img.cuda()
                maskgt = maskgt.cuda()
            optimizer.zero_grad()
            maskpred = net(img)                 
            loss_train_iou = loss_iou(maskpred, maskgt)            
            loss = loss_train_iou*args.loss_multiplier
            # loss.requires_grad =True 
            loss.backward()
            optimizer.step()
            # info
            progress(step,iters_per_epoch-1,epoch,str(loss.item()))
            # if step % args.disp_interval == 0:
                # print("[epoch %2d][iter %4d] loss: %.4f " \
                #                 % (epoch, step, loss))
        print("done training epoch %d"%(epoch))
        end = time.time()
        print('time elapsed: %fs' % (end - start))        
        if epoch%args.save_epoch==0 or epoch==args.max_epochs-1:
            if not os.path.exists(args.model_dir):
                        os.makedirs(args.model_dir)
            save_name = os.path.join(args.model_dir, 'saved_model_{}_{}.pth'.format(args.session, epoch))
            torch.save({'epoch': epoch+1, 'model': net.state_dict(), }, save_name)

            print('save model: {}'.format(save_name))
        print('evaluating...')
        eval_loss = 0
        with torch.no_grad():
            # setting to eval mode
            net.eval()
            eval_data_iter = iter(eval_dataloader)
            for i, data in enumerate(eval_data_iter):
                # print(i,'/',len(eval_data_iter)-1)
                progress(i,len(eval_data_iter)-1,epoch," iters")
                # data = eval_data_iter.next()  
                img,maskgt=data
                if args.cuda:
                    img = img.cuda()
                    maskgt = maskgt.cuda()            
                maskpred = net(img)
                loss_eval_iou = loss_iou(maskpred, maskgt)               
                eval_loss += loss_eval_iou *args.loss_multiplier
                if args.save_images and i%100==0:
                    if not os.path.exists(args.dir_images):
                        os.makedirs(args.dir_images)
                    # sample_mask = maskpred[0].cpu().detach().numpy()
                    numbere=f'{epoch:02d}'
                    numberi=f'{i:05d}'
                    filename = args.dir_images+'trainingpred_'+numbere+'_'+numberi+'.png'
                    filename = str(filename)
                    threshold = maskpred.mean()
                    imgmasked = img.clone()
                    maskpred3=maskpred.repeat(1,3,1,1)
                    imgmasked[maskpred3<=threshold]/=3
                    save_image(imgmasked[0], filename)           
                
            eval_loss = eval_loss/len(eval_dataloader)
            # val_loss_arr.append(eval_loss)
            print("eval done")
            print("[epoch %2d] loss: %.4f " \
                            % (epoch, torch.sqrt(eval_loss)))
            with open(os.path.join(args.model_dir, 'training_log_{}.txt'.format(args.session)), 'a') as f:
                f.write("[epoch %2d] loss: %.4f\n" \
                            % (epoch, torch.sqrt(eval_loss)))
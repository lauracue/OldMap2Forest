import argparse
import math
import os
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from src.metrics import evaluate_metrics
import random


from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    check_folder,
    plot_figures,
    read_tiff
)

from src.multicropdataset import RasterTileDataset
from src.models import ResUnet, WeightedFocalLoss, WeightedBCELoss
from src.models import WeightedDiceLoss, WeightedJaccardLoss, FBetaLoss



def main(parser, ar, ls, bal, opt):
    global args, figures_path
    args = parser.parse_args()
    
    args.arch = ar
    args.loss_fun = ls
    args.balance = bal
    args.opt = opt
    
    
    args.dump_path = args.dump_path + '/' + args.arch + '/' + '{}_{}_{}_{}'.format(args.size_crops,args.loss_fun,args.balance,args.opt)
    
    check_folder(args.dump_path)
    fix_random_seeds(args.seed[0])
    
    figures_path = os.path.join(args.dump_path, 'figures')
    check_folder(figures_path)

    
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    
    
    ######### Load Data ############
    
    tr_img = read_tiff('Overzichtskaart Semarang.tiff')
    tr_gt = read_tiff('train_mask.tif')
    tr_gt[tr_gt==0] = 3
    tr_gt[tr_gt==2] = 0
    tr_gt[tr_gt==3] = 2
    
    test_gt = read_tiff('test_mask.tif')
    test_gt[test_gt==0] = 3
    test_gt[test_gt==2] = 0
    test_gt[test_gt==3] = 2
    
    args.features = tr_img.shape[0]

    # get balanced coordinates
    coordinates, w0, w1 = get_coordinates(tr_gt, args.size_crops, args.balance)
    
    coordinates_val, w0, w1 = get_coordinates(test_gt, args.size_crops, args.balance)
    
    # define loader
    train_dataset = RasterTileDataset(img=tr_img, 
                                    lab=tr_gt, 
                                    coords=coordinates, 
                                    psize=args.size_crops,
                                    samples=args.samples)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    
    
    # define loader
    val_dataset = RasterTileDataset(img=tr_img, 
                                    lab=test_gt, 
                                    coords=coordinates_val, 
                                    psize=args.size_crops,
                                    samples=args.samples//5)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    
    
     
    logger.info("Building data done with {} images loaded.".format(len(train_loader)))
    
    # build model
    if args.arch == 'resunet':
        model = ResUnet(channel = args.features,
                        classes=1, 
                        filters=args.filters)
        

    # copy model to GPU
    model = model.cuda()
    
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.wd
        )
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.wd
        )
    elif args.opt == 'RAdam':
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.wd
        )
    
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")


    # optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_loss":0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]


    cudnn.benchmark = True

    best_loss = to_restore["best_loss"]
    
    activ = nn.Sigmoid()
    
    # training statistics
    global_losses = AverageMeter()    
    global_acc = AverageMeter()    
    global_f1 = AverageMeter()    
    global_pre = AverageMeter()    
    global_rec= AverageMeter()    
    global_tresh = AverageMeter()
    
    # val statistics
    acc_1 = AverageMeter()    
    f1_avg_post = AverageMeter()
    tresh_avg = AverageMeter()
    
    pre_avg_post = AverageMeter()
    rec_avg_post = AverageMeter()
        
    if args.loss_fun == "focal":
        criteria = [WeightedFocalLoss()]
    elif args.loss_fun == "f1beta":
        criteria = [FBetaLoss(beta=1)]
    elif args.loss_fun == "cross":
        criteria = [WeightedBCELoss()]
    elif args.loss_fun == 'dice':
        criteria = [WeightedDiceLoss()]
    elif args.loss_fun == 'jaccard':
        criteria = [WeightedJaccardLoss()]
    if args.loss_fun == "focal+cross":
        criteria = [WeightedFocalLoss(),WeightedBCELoss()]
    elif args.loss_fun == "dice+cross":
        criteria = [WeightedDiceLoss(),WeightedBCELoss()]
    elif args.loss_fun == 'focal+dice':
        criteria = [WeightedDiceLoss(),WeightedFocalLoss()]
    
    if best_loss > 0.:
        global_losses.update(best_loss, 1)
        
        
    patience = 50

    for epoch in range(start_epoch, args.epochs):
        
        model.train()
        
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)
        

        for it, (inp_img, ref) in enumerate(train_loader):      
    
            # update learning rate
            iteration = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]
    
            # ============ forward pass and loss ... ============
            # compute model loss and output
            inp_img = inp_img.cuda(non_blocking=True)
            ref = ref.cuda(non_blocking=True)
            
            mask_ref = torch.ones(ref.shape)
            mask_ref[ref==2] = 0
            
            if args.balance == 2:
                weight_mask = torch.zeros(ref.shape)
                weight_mask[ref==0] = w0
                weight_mask[ref==1] = w1
                weight_mask = weight_mask.cuda(non_blocking=True)
                
            else:
                weight_mask = torch.ones(ref.shape)
                weight_mask = weight_mask.cuda(non_blocking=True)
                            
            
            # calculate losses
            out_batch = model(inp_img)
            
            
            # ============ model loss ... ============
            if len(criteria) == 1:
                loss = criteria[0](out_batch[:,0,:,:][mask_ref==1],ref[mask_ref==1],weight_mask[mask_ref==1])
            else:
                loss = criteria[0](out_batch[:,0,:,:][mask_ref==1],ref[mask_ref==1],weight_mask[mask_ref==1]) 
                loss+=criteria[1](out_batch[:,0,:,:][mask_ref==1],ref[mask_ref==1],weight_mask[mask_ref==1])

            
            # ============ backward and optim step ... ============
            optimizer.zero_grad()
            loss.backward()
            
            # performs updates using calculated gradients
            optimizer.step()
            
            # update the average loss
            global_losses.update(loss.item())
    
            # Evaluate summaries only once in a while
            if it % 50 == 0:
                summary_batch=evaluate_metrics(activ(out_batch[:,0,:,:][mask_ref==1]), ref[mask_ref==1])
                
                global_acc.update(torch.tensor([summary_batch["Accuracy"]]), inp_img.size(0))
                global_f1.update(torch.tensor([summary_batch["F0.5"]]), inp_img.size(0))
                global_pre.update(torch.tensor([summary_batch["Pre"][1]]), inp_img.size(0))
                global_rec.update(torch.tensor([summary_batch["Rec"][1]]), inp_img.size(0))
                global_tresh.update(torch.tensor([summary_batch["Best Threshold"]]), inp_img.size(0))
                
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc {acc.val[0]:.2f} ({acc.avg[0]:.2f})\t"
                    "F0.5 {f1.val[0]:.2f} ({f1.avg[0]:.2f})\t"
                    "Pre {pre.val[0]:.2f} ({pre.avg[0]:.2f})\t"
                    "Rec {rec.val[0]:.2f} ({rec.avg[0]:.2f})\t"
                    "Tresh {tresh.val[0]:.2f} ({tresh.avg[0]:.2f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        it,
                        loss=global_losses,
                        acc = global_acc,
                        f1 = global_f1,
                        pre = global_pre,
                        rec = global_rec,
                        tresh = global_tresh,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
                # logger.info(summary_batch)
                
            if it == 0:
                # plot samples results for visual inspection
                plot_figures(inp_img,ref,activ(out_batch[:,0,:,:]),figures_path,epoch,'train')
                
        
        training_stats.update((epoch, global_losses.avg))
        
        ######################### start validation #############
        model.eval()
    
        for it, (inp_img, ref) in enumerate(val_loader):      
    
            # ============ forward pass and loss ... ============
            # compute model loss and output
            inp_img = inp_img.cuda(non_blocking=True)
            # inp_img = torch.nan_to_num(inp_img, nan=0.0, posinf=0.0)
            ref = ref.cuda(non_blocking=True)
            
            mask_ref = torch.ones(ref.shape)
            mask_ref[ref==2] = 0
            
            
            # calculate losses
            out_batch = model(inp_img)
        
            summary_batch=summary_batch=evaluate_metrics(activ(out_batch[:,0,:,:][mask_ref==1]), ref[mask_ref==1])
            
            acc_1.update(torch.tensor([summary_batch["Accuracy"]]), inp_img.size(0))
    
            f1_avg_post.update(torch.tensor([summary_batch["F0.5"]]), inp_img.size(0))
            
            tresh_avg.update(torch.tensor([summary_batch["Best Threshold"]]), inp_img.size(0))
            
            pre_avg_post.update(torch.tensor([summary_batch["Pre"][1]]), inp_img.size(0))
            
            rec_avg_post.update(torch.tensor([summary_batch["Rec"][1]]), inp_img.size(0))
            
    
        logger.info("============ Validation metrcis ... ============")
        logger.info(
            "Tresh ({tresh_avg.avg[0]:.3f})\t"
            "Pre positive ({pre_avg_post.avg[0]:.2f})\t"
            "Rec positive ({rec_avg_post.avg[0]:.2f})\t"
            "F0.5 ({f1_avg_post.avg[0]:.2f})\t"
            "Acc ({acc_1.avg[0]:.2f})".format(
                pre_avg_post = pre_avg_post,
                rec_avg_post = rec_avg_post,
                tresh_avg = tresh_avg,
                f1_avg_post = f1_avg_post,
                acc_1=acc_1,
            )
        )        
    
        # plot samples results for visual inspection
        plot_figures(inp_img,ref,activ(out_batch[:,0,:,:]),figures_path,epoch,'val')
        

        # save checkpoints
        sve_ind = True if np.round(f1_avg_post.avg.data.cpu().numpy()[0],4) > best_loss else False
        if sve_ind:
            logger.info("Savig best model at epoch {}".format(epoch))
            best_loss = np.round(f1_avg_post.avg.data.cpu().numpy()[0],4)
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
                
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "bestcheckpoint.pth.tar"),
            )
            
            patience=50
            
        else:
            patience-=1
            
            
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
            
        torch.save(
            save_dict,
            os.path.join(args.dump_path, "checkpoint.pth.tar"),
        )   
        
        # if patience==0:
        #     logger.info("============ Early stop after 50 epochs with no improvement ... ============")
        #     break
        




def get_coordinates(gt, size_crops, balance=0):

    total_zeros = 0
    total_ones = 0

    gt[:size_crops, :] = 2
    gt[-size_crops:, :] = 2
    gt[:, :size_crops] = 2
    gt[:, -size_crops:] = 2


    # Coordinates of zeros and ones
    zeros = np.where((gt == 0))
    ones = np.where((gt == 1))
    zeros_coordinates = list(zip(zeros[0], zeros[1]))
    ones_coordinates = list(zip(ones[0], ones[1]))

    total_zeros += len(zeros_coordinates)
    total_ones += len(ones_coordinates)

    if balance > 0:
        min_samples = min(len(zeros_coordinates), len(ones_coordinates))
        balanced_zeros = random.sample(zeros_coordinates, min_samples)
        balanced_ones = random.sample(ones_coordinates, min_samples)
    else:
        max_samples = max(len(zeros_coordinates)//10, len(ones_coordinates))
        
        
        # Randomly sample min_samples from both zeros and ones
        balanced_zeros = random.sample(zeros_coordinates, max_samples)
        balanced_ones = random.sample(ones_coordinates, len(ones_coordinates))
        
        
    # Combine and shuffle coordinates
    balanced_zeros = np.asarray(balanced_zeros)
    balanced_ones = np.asarray(balanced_ones)
    
    combined_coords = np.vstack((balanced_zeros,balanced_ones))
    

    # Calculate weights
    if total_zeros == 0 or total_ones == 0:
        w0 = w1 = 1
    else:
        w0 = (total_zeros + total_ones) / (2 * total_zeros)
        w1 = (total_zeros + total_ones) / (2 * total_ones)

    return combined_coords, w0, w1



if __name__ == "__main__":
    
    logger = getLogger('resunet_model')
    
    parser = argparse.ArgumentParser(description="Training of ResUnet")
    
    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--dump_path", type=str, default="./exp/v2",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--samples", type=int, default=5000, 
                        help="nb samples per epoch")
    parser.add_argument("--size_crops", type=int, default=128, 
                        help="Size of the input tile for the network")
    parser.add_argument("--features", type=int, default=3, 
                        help="Number of features")
                        
    
    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--arch", default="resunet", type=str, 
                        help="convnet architecture --> 'resunet','deeplabv3+', deeplabv3resnet50")
    parser.add_argument("--pretrained", default=True, type=bool, 
                        help="True for load pretrained weights from Imagenet")
    parser.add_argument("--filters", default=[16,16,16,16], type=int, 
                        help="Filter for the ResUnet for trained from scratch")
    parser.add_argument("--loss_fun", default="dice+cross", type=str, 
                        help="cross, focal, jaccard, dice, asymmetric, dice+cross, focal+cross")
    parser.add_argument("--balance", type=int, default=1, 
                        help="0 not balance at all, 1 balance coordinates, 3 balance coords + add weight")
    parser.add_argument("--opt", default='RAdam', type=str, 
                        help="Optimizer, SGD, Adam or RAdam")
    
    
    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=50, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=0.01, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0.000001, help="final learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    
    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    
    ##########################
    #### others parameters ###
    ##########################
    
    parser.add_argument("--workers", default=6, type=int,
                        help="number of data loading workers")
    parser.add_argument("--seed", type=int, default=[31], help="seeds")
    
    
    arch = ['resunet']
    loss_fun = ["dice+cross"]
    # loss_fun = ['dice',"focal+dice","focal+cross","dice+cross"]
    balance = [1]  # 0 not balance at all, 1 balance coordinates, 2 balance coords + add weight
    optimizer = ['RAdam']
    for ar in arch:
        for ls in loss_fun:
            for bal in balance:
                for opt in optimizer:
                    main(parser, ar, ls, bal, opt)
                    
                    
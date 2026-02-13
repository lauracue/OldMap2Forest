# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from osgeo import gdal

from src.logger import create_logger


from src.utils import (
    bool_flag,
    check_folder,
    extract_patches_coord, 
    array2raster,
    read_tiff
)

from src.models import ResUnet
from src.multicropdataset import DataloaderEval


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(ov, ar, ls, bal, opt):
    global args, figures_path
    args = parser.parse_args()
    
    args.arch = ar
    args.loss_fun = ls
    args.balance = bal
    args.opt = opt
    
    args.overlap = ov
    
    args.dump_path = args.dump_path + '/' + args.arch + '/' + '{}_{}_{}_{}'.format(args.size_crops,args.loss_fun,args.balance,args.opt)
    
    args.pretrained = os.path.join(args.dump_path,args.pretrained)
    

    ######## get data #####

    
    # create a logger
    logger = create_logger(os.path.join(args.dump_path, "inference.log"),rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    test_img = read_tiff('Overzichtskaart Semarang.tiff')
        
    ch, row, col = test_img.shape
    
    mask = np.ones((row, col))
    
    coords, stride = extract_patches_coord(mask, args.size_crops, args.overlap)

    
    pred_prob = np.zeros(shape = (row, col), dtype='float32')
    
    # build model
    model = ResUnet(channel = ch,
                    classes=1, 
                    filters=args.filters)
                   
    
            
    # model to gpu
    model = model.cuda()
    model.eval()
    
    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:0", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    cudnn.benchmark = True

    check_folder(os.path.join(args.dump_path,'prediction'))


    # define loader
    val_dataset = DataloaderEval(test_img,
                                coords,
                                psize = args.size_crops)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
        
    logger.info("Building data done")
    
    pred_prob = predict_network(val_loader, model, coords, pred_prob, logger)
        
    raster_src = gdal.Open('train_mask.tif')
    
    array2raster(os.path.join(args.dump_path,'prediction',f'prob_map_{args.overlap}.tiff'), raster_src, pred_prob, "Float32")

    logger.info("============ Inference finished ============")
    

def predict_network(dataloader, model, coords, pred_prob, logger):
    model.eval()
    sig = nn.Sigmoid()

    psize = args.size_crops
    half = psize // 2
    keep_size = int(psize * (1 - args.overlap))
    trim = (psize - keep_size) // 2

    pred_prob_t = torch.from_numpy(pred_prob).cuda(non_blocking=True)

    j = 0
    with torch.no_grad():
        for inputs in dataloader:

            x1 = inputs.cuda(non_blocking=True)

            out = model(x1)

            out = sig(out)[:, 0]                     # (B, H, W)
            core = out[:,                            # slice once
                       trim:trim + keep_size,
                       trim:trim + keep_size]

            bsz = core.shape[0]
            coord_x = coords[j:j + bsz, 0]
            coord_y = coords[j:j + bsz, 1]

            for b in range(bsz):
                cx = coord_x[b]
                cy = coord_y[b]

                r0 = cx - half + trim
                c0 = cy - half + trim

                pred_prob_t[
                    r0:r0 + keep_size,
                    c0:c0 + keep_size
                ] = core[b]

            j += bsz

    return pred_prob_t.cpu().numpy()

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of FCN")
    
    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--dump_path", type=str, default="./exp/v2",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--size_crops", type=int, default=128, 
                        help="Size of the input tile for the network")
    parser.add_argument("--overlap", type=float, default=[0.2], 
                        help="samples per epoch")
    

    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--arch", default="resunet", type=str, 
                        help="convnet architecture --> 'resunet','deeplabv3+'")
    parser.add_argument("--filters", default=[16,16,16,16], type=int, 
                        help="Filter for the ResUnet for trained from scratch")

    
    
    ##########################
    #### others parameters ###
    ##########################
    parser.add_argument("--workers", default=6, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=1,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=bool_flag, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    
    
    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--pretrained", default="checkpoint.pth.tar", type=str, 
                        help="path to pretrained weights")
    
    
    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size ")

    
    
    overlaps = [0.2]
    arch = ['resunet']
    loss_fun = ["dice+cross"]
    # loss_fun = ['dice',"focal+dice","focal+cross","dice+cross"]
    balance = [1]  # not balance at all, 1 balance coordinates, 3 balance coords + add weight
    optimizer = ['RAdam']

    for ov in overlaps:
        for ar in arch:
            for ls in loss_fun:
                for bal in balance:
                    for opt in optimizer:
                        main(ov, ar, ls, bal, opt)

import datetime
import os
import time

import torch
from torchvision.utils import make_grid, save_image
import torch.utils.data
from torch import nn
import warnings
warnings.filterwarnings("ignore")

from models_refer.model import VPDRefer

import transforms as T
import utils
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPTextModel
import gc
from collections import OrderedDict

# Adding control path to script
import sys
sys.path.append('../ControlNet')


def get_dataset(image_set, transform, args):
    from data.dataset_refer_clip import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=(image_set == 'val')
                      )
    num_classes = 2

    return ds, num_classes

def get_dataset_control(image_set, transform, args):
    from data.dataset_refer_clip import ReferDatasetControl
    ds = ReferDatasetControl(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                  ]

    return T.Compose(transforms)


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader, clip_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, hint = data
            image, target, sentences, attentions, hint = image.cuda(non_blocking=True), target.cuda(non_blocking=True), \
                                                         sentences.cuda(non_blocking=True), attentions.cuda(non_blocking=True), \
                                                         hint.cuda(non_blocking=True)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for idx in range(sentences.size(-1)):
                
                embedding = clip_model(input_ids=sentences[:, :, idx]).last_hidden_state
                attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

                # Added hint // August
                output = model(image, embedding, hint=hint)

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                acc_ious += this_iou
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

                if idx == 0:  # Save output image // August
                    img_unnorm = torch.clamp(image[0] / 2 + 0.5, min=0, max=1).cpu()  # Unnormalize // August
                    img_output = torch.tile(output.argmax(1)[0], dims=(3,1,1))
                    img_target = torch.tile(torch.tensor(target[0]), dims=(3,1,1))
                    img_hint = hint[0].cpu()
                    img_list = [img_unnorm, img_hint, img_output, img_target]
                    row = torch.stack(img_list)
                    grid_img = make_grid(row, nrow=len(row), padding=4)
                    file_name = '../saved_images/train_run0/bbox_model_ite_%d.png' % total_its
                    save_image(grid_img, file_name)



        
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq, iterations, clip_model):

    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory/10**9  # Used to gauge model footprint // August

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    #for data in metric_logger.log_every(data_loader, print_freq, header):
    for data in data_loader:
        total_its += 1
        print("Iteration {}/{}".format(total_its, len(data_loader)))
        image, target, sentences, attentions, hint = data
        image, target, sentences, attentions, hint = image.cuda(non_blocking=True),\
                                                     target.cuda(non_blocking=True),\
                                                     sentences.cuda(non_blocking=True),\
                                                     attentions.cuda(non_blocking=True), \
                                                     hint.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        
        #embedding = clip_model(input_ids=sentences).last_hidden_state
        embedding = clip_model(input_ids=sentences[:,:,0]).last_hidden_state  # clip_model only works what i assume to be one sentence at a time // August
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        # Try with dummy hint, this needs to come from the dataloader // August
        # Notice that the hint is parsed in image space and not latent space // August
        #hint = torch.zeros([1, 3, 512, 512]).to(device='cuda')
        # Try with None hint, to see if model can still handle this // August
        #hint = None

        output = model(image, embedding, hint=hint)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        #torch.cuda.synchronize  # Disable synchronize // August
        train_loss += loss.item()
        iterations += 1
        #metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        del embedding

        gc.collect()
        torch.cuda.empty_cache()
        #torch.cuda.synchronize()  # Disable synchronize // August

    # Print max memory reserved during epoch // August
    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved during training step: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Added main_single_process
def main_single_process(args):

    # Define datasets
    dataset, num_classes = get_dataset_control("train", get_transform(args=args), args=args)
    dataset_test, _ = get_dataset_control("val", get_transform(args=args), args=args)

    n_train = len(dataset)
    n_test = len(dataset_test)
    print("Number of samples in train dataset: %d" % n_train)
    print("Number of samples in test dataset: %d" % n_test)

    # Define dataset subsets for train and test
    subset_size_train = 4000  # Number of samples in train subset // August
    subset_size_test = 200  # Number of samples in test subset // August
    subset_indices_train = torch.randperm(len(dataset))[:subset_size_train]
    subset_indices_test = torch.randperm(len(dataset_test))[:subset_size_test]
    subset_train = torch.utils.data.Subset(dataset, indices=subset_indices_train)
    subset_test = torch.utils.data.Subset(dataset_test, indices=subset_indices_test)
    dataset = subset_train  # Subset works, but only with batch_size = 1 // August
    dataset_test = subset_test  # Subset works, but only with batch_size // August

    # Disable samplers related to ddp // August
    # batch sampler
    #print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    #num_tasks = utils.get_world_size()
    #global_rank = utils.get_rank()
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    #test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # data loader - single process // August
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=args.workers)

    print("Creating subset of train dataloader %d / %d" % (subset_size_train, n_train))
    print("Creating subset of test dataloader %d / %d" % (subset_size_test, n_test))

    model = VPDRefer(sd_path='../checkpoints/v1-5-pruned-emaonly.ckpt', neck_dim=[320,640+args.token_length,1280+args.token_length,1280], use_original_vpd=args.use_original_vpd, controlnet_batch_size=args.batch_size)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # Disable ddp here // August
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    #single_model = model.module

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.cuda()
    clip_model = clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        #single_model.load_state_dict(checkpoint['model'])
        model.load_state_dict(checkpoint['model'], strict=False)  # Changed to model and added non-strict // August


    # parameters to optimize
    lesslr_no_decay = list()
    lesslr_decay = list()
    no_lr = list()
    no_decay = list()
    decay = list()
    #for name, m in single_model.named_parameters():
    for name, m in model.named_parameters():  # Changed single_model -> model // August
        if 'unet' in name and 'norm' in name:
            lesslr_no_decay.append(m)
        elif 'unet' in name:
            lesslr_decay.append(m)
        elif 'encoder_vq' in name:
            no_lr.append(m)
        elif 'norm' in name:
            no_decay.append(m)
        else:
            decay.append(m)

    params_to_optimize = [
        {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale':0.01},
        {'params': lesslr_decay, 'lr_scale': 0.01},
        {'params': no_lr, 'lr_scale': 0.0},
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay}
    ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        # Disabling these lines as checkpoint only contains key 'model' // August
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # resume_epoch = checkpoint['epoch']
        resume_epoch = -999
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        print("Epoch {}/{}".format(epoch+1, args.epochs))
        #data_loader.sampler.set_epoch(epoch)  # data_loader has no sampler // August

        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, iterations, clip_model)

        iou, overallIoU = evaluate(model, data_loader_test, clip_model)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            # dict_to_save = {'model': single_model.state_dict(),
            #                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
            #                'lr_scheduler': lr_scheduler.state_dict()}

            dict_to_save = {'model': model.state_dict(),  # Changed single_model -> model // August
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    # Add default args // August
    # example: bash train.sh refcoco /path/to/logdir <NUM_GPUS> --token_length 40
    '''
    --master_port 12345 train.py \
    --dataset $1 --model_id $1 \
    --batch-size 4 --lr 0.00005 --wd 1e-2 \
    --epochs 40 --img_size 512 ${@:4} \
    2>&1 | tee $logdir/log.txt
    '''
    args.disable_ddp = True  # Trying to remove multi-GPU training // August
    args.master_port = 12345
    args.batch_size = 1  # Batch size for training
    args.workers = 1
    args.pin_mem = False
    args.nproc_per_node = 1
    args.lr = 0.00005
    args.wd = 1e-2
    args.epochs = 10
    args.token_length = 40
    args.dataset = "refcoco"
    args.model_id = "refcoco"
    args.img_size = 512  # 512 (original)
    # Override arg with path to vpd pre-trained weights // August
    args.resume = "../saved_models/vpd_ris_refcoco.pth"
    args.output_dir = "../saved_models"
    args.use_original_vpd = False

    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    print('Batch size for training: %d' % args.batch_size)
    # main(args)
    main_single_process(args)  # Change to run on single GPU // August

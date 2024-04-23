import datetime
import os
import time
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn

import transforms as T
import utils
from transformers.models.clip.modeling_clip import CLIPTextModel
from models_refer.model import VPDRefer
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Adding control path to script
import sys
sys.path.append('../ControlNet')

def get_dataset(image_set, transform, args):
    from data.dataset_refer_clip import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, dataset_test, data_loader, clip_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for idx in range(sentences.size(-1)):
            #for idx in range(10):  # Testing fewer inferences // August

                embedding = clip_model(input_ids=sentences[:, :, idx]).last_hidden_state
                attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
                print("Input image shape:", image.shape)
                print("Input embedding shape:", embedding.shape)
                print("sentences[idx]", sentences[:, :, idx])

                # Try with dummy hint, this needs to come from the dataloader // August
                # Notice that the hint is parsed in image space and not latent space // August
                hint = torch.zeros([1, 3, 512, 512]).to(device='cuda')
                # Try with None hint, to see if model can still handle this // August
                hint = None

                output = model(image, embedding, hint=hint)
                output = output.cpu()

                print("Output shape:", output.shape)
                plt.figure(figsize=(30,10))
                plt.subplot(1, 3, 1)
                plt.imshow(image[0].permute(1, 2, 0).cpu())
                #cvt_sentence = dataset_test.tokenizer.convert_tokens_to_string(sentences[:, :, idx])
                #plt.title(cvt_sentence)
                plt.subplot(1, 3, 2)
                plt.imshow(output[0, 0, :, :])
                plt.subplot(1, 3, 3)
                plt.imshow(output[0, 1, :, :])
                plt.show()
                input("Press Enter to continue...")

                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

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


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    # Add default args // August
    # example: refcoco /path/to/vpd_ris_refcoco.pth --token_length 40
    args.token_length = 40
    args.dataset = "refcoco"
    args.img_size = 512  # 512 (original)
    # Override arg with path to vpd pre-trained weights // August
    args.resume = "../saved_models/vpd_ris_refcoco.pth"

    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    # print(args.model)
    
    single_model = VPDRefer(sd_path='../checkpoints/v1-5-pruned-emaonly.ckpt',
                      neck_dim=[320,640+args.token_length,1280+args.token_length,1280],
                      use_original_vpd=False)

    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.cuda()
    clip_model = clip_model.eval()

    test_controlnet_dataset = False
    if test_controlnet_dataset:
        #### Add controlnet dataset // August
        print("Adding ControlNet")
        from ControlNet.tutorial_dataset import MyDataset
        controlnet_dataset = MyDataset()

        ### Test MyDataset // August
        print(len(controlnet_dataset))

        item = controlnet_dataset[1234]
        jpg = item['jpg']
        txt = item['txt']
        hint = item['hint']
        print(txt)
        print(jpg.shape)
        print(hint.shape)

    ### Evaluate VPD
    evaluate(model, dataset_test, data_loader_test, clip_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)

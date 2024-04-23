import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from ldm.util import instantiate_from_config

from omegaconf import OmegaConf
from lib.mask_predictor import SimpleDecoding

from vpd.models import UNetWrapper, TextAdapterRefer, ControlNetWrapper

### Fails to import vpd.models as vpd module cannot be found ### August



class VPDRefer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 sd_path=None,
                 base_size=512,
                 token_embed_dim=768,
                 neck_dim=[320,680,1320,1280],
                 use_original_vpd=False,
                 **args):
        super().__init__()

        # Define flag to disable/enable original VPD
        self.use_original_vpd = use_original_vpd

        if self.use_original_vpd:
            config = OmegaConf.load('./v1-inference.yaml')
            config.model.params.ckpt_path = f'{sd_path}'
            # import pdb; pdb.set_trace()
            sd_model = instantiate_from_config(config.model)
            # This is the encoder part of the StableDiffusion model - this needs to be replaced with part of ControlNet
            self.encoder_vq = sd_model.first_stage_model

            # Original code here
            self.unet = UNetWrapper(sd_model.model, base_size=base_size)
            del sd_model.cond_stage_model
            del self.encoder_vq.decoder
        else:
            # Testing adding of ControlLDM
            print("ControlLDM")
            #control_config = OmegaConf.load('../ControlNet/models/cldm_v15.yaml')
            from ControlNet.cldm import cldm
            from ControlNet.cldm import model
            from ControlNet.cldm.model import load_state_dict
            self.controlnet = model.create_model('../ControlNet/models/cldm_v15.yaml')
            print("ControlNET initialized successfully")

            resume_path = "../ControlNet/models/control_sd15_ini.ckpt"
            self.controlnet.load_state_dict(load_state_dict(resume_path, location='cpu'))
            self.controlnet.batch_size = 1
            self.controlnet.logger_freq = 300
            self.controlnet.learning_rate = 1e-5
            self.controlnet.sd_locked = True
            self.controlnet.only_mid_control = False

            self.controlwrapper = ControlNetWrapper(self.controlnet, base_size=base_size)
            print("VPDRefer initialized successfully")

        self.text_adapter = TextAdapterRefer(text_dim=token_embed_dim)
        self.classifier = SimpleDecoding(dims=neck_dim)
        self.gamma = nn.Parameter(torch.ones(token_embed_dim) * 1e-4)

    def forward(self, img, l_feats, hint=None):
        """
        I assume they first parse the input "img" to the latent space encoder to obtain latent vectors, then these latent vectors
        are parsed through a text_adapter, which returns cross-attention between the latent vectors of the
        images and the latent features of the text encoder - this is done as according to the authors,
        "The cross-attention map between the feature map and the conditioning text feature enjoys good locality"
        I assume the parameter l_feats are the features from the text encoder, as they are parsed directly from the
        dataloader. // August

        the UNetWrapper (self.unet) is designed to output 4 cross-attention maps, 1 for each resolution apart from the
        last 8x8 block.
        """

        input_shape = img.shape[-2:]

        if self.use_original_vpd:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img).mode().detach()  # Original
            c_crossattn = self.text_adapter(latents, l_feats, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
            t = torch.ones((img.shape[0],), device=img.device).long()
            outs = self.unet(latents, t, c_crossattn=[c_crossattn])  # Disabled StableDiffusion model // August
        else:
            # Test here outputs of controlwrapper
            c_concat = hint  # Assign hint here // August
            with torch.no_grad():
                latents = self.controlnet.encode_first_stage(img).mode().detach()
            c_crossattn = self.text_adapter(latents, l_feats, self.gamma)
            t = torch.ones((img.shape[0],), device=img.device).long()
            # In controlwrapper, c_concat is interpreted as the hint
            outs = self.controlwrapper(latents, t, c_crossattn=[c_crossattn], c_concat=[c_concat])

        # Run the VPD head (red box)
        x_c1, x_c2, x_c3, x_c4 = outs  # Replaced by output from ControlNet (same as ControlNet) // August
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x

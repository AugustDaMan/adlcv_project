from omegaconf import OmegaConf

import torch as th
import torch
import math
import abc

from torch import nn, einsum

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextModel, CLIPTextTransformer, _expand_mask
from inspect import isfunction


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            
            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2, is_cross, place_in_unet)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0

    import ldm.models.diffusion.ddpm  # Enables original code to run
    if type(model) == ldm.models.diffusion.ddpm.DiffusionWrapper:  # Enables original code to run
        sub_nets = model.diffusion_model.named_children()
    else:
        # Testing fetching the correct model class from ControlLDM // August
        # ControlLDM
        #    control_model (ControlNet)
        #    model (DiffusionWrapper)
        sub_nets = model.control_model.named_children()


    for net in sub_nets:
        if "input_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def register_attention_control_ControlNet(model, controller):
    # This function should be used to register the attention outputs of the ControlNet model // August
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2, is_cross, place_in_unet)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0

    # Testing fetching the correct model class from ControlLDM // August
    # ControlLDM
    #    control_model (ControlNet)
    #    model (DiffusionWrapper)

    # Attempting to use ControlledUnetModel from cldm as the base for registering intermediate features.
    # This model has input_blocks, middle_blocks and output_block same as UnetModel from StableDiffusion,
    # however, unsure if this is equivalent to extracting features from the ControlNet encoder. It could
    # also just be that we are extracting features from the StableDiffusion encoder which I don't think is correct.
    sub_nets = model.model.diffusion_model.named_children()  # Original StableDiffusion  // August
    sub_nets_control = model.model.diffusion_model.named_children()  # model.control_model.named_children()  # ControlNet  // August

    for net_control, net in zip(sub_nets_control, sub_nets):
        # I assume we need to keep the attention outputs of the ControlNet encoder,
        # but keep the attention outputs of the regular StableDiffusion decoder  // August
        # Not sure what to use middle_hint and input_hint blocks for in ControlNet? // August

        # input blocks use
        if "input_blocks" in net_control[0]:
            cross_att_count += register_recr(net_control[1], 0, "down")

        # output blocks use
        elif "output_blocks" in net_control[0]:
            cross_att_count += register_recr(net_control[1], 0, "up")

        # middle blocks use
        elif "middle_block" in net_control[0]:
            cross_att_count += register_recr(net_control[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.max_size) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, base_size=64, max_size=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.base_size = base_size
        if max_size is None:
            self.max_size = self.base_size // 2
        else:
            self.max_size = max_size

def register_hier_output(model):
    self = model.diffusion_model
    from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding
    def forward(x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # import pdb; pdb.set_trace()
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        out_list = []

        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)
        h = h.type(x.dtype)

        out_list.append(h)
        return out_list
    
    self.forward = forward

class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=True, base_size=512, max_attn_size=None, attn_selector='up_cross+down_cross') -> None:
        super().__init__()
        self.unet = unet
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size16 = base_size // 32  # 32
        self.size32 = base_size // 16  # 16
        self.size64 = base_size // 8   # 8
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        register_hier_output(unet)
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        if self.use_attn:
            self.attention_store.reset()
        out_list = self.unet(*args, **kwargs)
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn)
            out_list[1] = torch.cat([out_list[1], attn16], dim=1)
            out_list[2] = torch.cat([out_list[2], attn32], dim=1)
            if attn64 is not None:
                out_list[3] = torch.cat([out_list[3], attn64], dim=1)
        return out_list[::-1]

    def process_attn(self, avg_attn):
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                size = int(math.sqrt(up_attn.shape[1]))
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        if len(attns[self.size64]) > 0:
            attn64 = torch.stack(attns[self.size64]).mean(0)
        else:
            attn64 = None
        return attn16, attn32, attn64

class ControlNetWrapper(nn.Module):
    """
    Wrapper class for ControlNet which should replace UNetWrapper class used in VPDRefer.
    This is a copy of UNetWrapper, but with some modifications. // August
    """

    def __init__(self, unet, use_attn=True, base_size=512, max_attn_size=None, attn_selector='up_cross+down_cross') -> None:
        super().__init__()
        self.unet = unet
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size16 = base_size // 32  # 32
        self.size32 = base_size // 16  # 16
        self.size64 = base_size // 8   # 8
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control_ControlNet(unet, self.attention_store)
        register_hier_output_ControlNet(unet.model)  # Trying adding .model here // August
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        # Assigning variables as sanity check // August
        latents = args[0]
        t = args[1]
        cond = kwargs

        if self.use_attn:
            self.attention_store.reset()
        # self.unet I believe should be of type UNetModel // August
        # out_list = self.unet(*args, **kwargs)  # Original

        # Testing hacking in of forward on ControlNet here
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # cond['c_concat'] should be a list on length 1 containing the hint image batch before encoding, which for the
        # case of tutorial_train is a tensor of shape [4, 3, 512, 512] // August
        if None in cond['c_concat']:  # Check that hint is not None
            out_list = self.unet.model.diffusion_model(x=latents,
                                                       timesteps=t,
                                                       context=cond_txt,
                                                       control=None,
                                                       only_mid_control=self.unet.only_mid_control)  # Runs forward on ControlledUnetModel w.o hint // August
        else:
            hint = torch.cat(cond['c_concat'], 1)
            # In control_model (class ControlNet) x is expected to be in latent space (4x64x64)
            # however, the hint is expected to be in image space (3x512x512) // August
            control = self.unet.control_model(x=latents, hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.unet.control_scales)]

            out_list = self.unet.model.diffusion_model(x=latents,
                                                       timesteps=t,
                                                       context=cond_txt,
                                                       control=control,
                                                       only_mid_control=self.unet.only_mid_control)  # Runs forward on ControlledUnetModel w. hint // August

        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn)
            out_list[1] = torch.cat([out_list[1], attn16], dim=1)
            out_list[2] = torch.cat([out_list[2], attn32], dim=1)
            if attn64 is not None:
                out_list[3] = torch.cat([out_list[3], attn64], dim=1)
        return out_list[::-1]

    def process_attn(self, avg_attn):
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                size = int(math.sqrt(up_attn.shape[1]))
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        if len(attns[self.size64]) > 0:
            attn64 = torch.stack(attns[self.size64]).mean(0)
        else:
            attn64 = None
        return attn16, attn32, attn64


def register_hier_output_ControlNet(model):
    # This functions assumes self is of type UNetModel (as in ControlledUnetModel(UNetModel) inside cldm.py)
    # The function is modified to parse the extra condition which is expected by ControlledUnetModel // August
    # This function overrides the forward function of UNetModel, which is why it needs to be changed.
    self = model.diffusion_model
    from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding
    def forward(x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        out_list = []  # // August

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

            if i in [1, 4, 7]:  # // August
                out_list.append(h)  # // August

        h = h.type(x.dtype)

        out_list.append(h)  # // August
        return out_list  # // August

    self.forward = forward

class TextAdapter(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = text_dim
        self.fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim)
        )

    def forward(self, latents, texts, gamma):
        n_class, channel = texts.shape
        bs = latents.shape[0]

        texts_after = self.fc(texts)
        texts = texts + gamma * texts_after
        texts = repeat(texts, 'n c -> b n c', b=bs)
        return texts
    
class TextAdapterRefer(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, latents, texts, gamma):
        texts_after = self.fc(texts)
        texts = texts + gamma * texts_after
        return texts


class TextAdapterDepth(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, latents, texts, gamma):
        # use the gamma to blend
        n_sen, channel = texts.shape
        bs = latents.shape[0]

        texts_after = self.fc(texts)
        texts = texts + gamma * texts_after
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, pool=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

        self.pool = pool

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        if self.pool:
            z = outputs.pooler_output
        else:
            z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
    
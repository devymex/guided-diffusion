import torch, torch.nn as nn
from guided_diffusion.unet import UNetModel

torch.set_printoptions(precision=8, sci_mode=False)

image_size = 512
time_range = 1000
chkpt_file = 'log/train/1679383593/model550000.pt'

class Diffusion(nn.Module):
    def __init__(self, image_size, time_range, chkpt_file):
        super().__init__()
        self.img_shape = [1, 3, image_size, image_size]
        self.time_range = time_range

        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, time_range)
        alphas = 1. - betas
        alcps = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', alphas, persistent=False)
        self.register_buffer('alcps', alcps, persistent=False)

        self.unet = UNetModel(
                image_size=image_size,
                in_channels=3,
                model_channels=128,
                out_channels=3,
                num_res_blocks=2,
                attention_resolutions=(32, 64),
                dropout=0.0,
                channel_mult=(0.5, 1, 1, 2, 2, 4, 4),
                num_classes=None,
                use_checkpoint=False,
                use_fp16=False,
                num_heads=4,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                resblock_updown=False,
                use_new_attention_order=False,
            )
        chkpt = torch.load(chkpt_file, map_location="cpu")
        self.unet.load_state_dict(chkpt)
        self.unet.eval()

    def forward(self):
        x_t = torch.randn(self.img_shape, device=self.betas.device)
        for t in reversed(range(time_range)):
            with torch.no_grad():
                eps = self.unet(x_t, torch.tensor([t], dtype=torch.int64, device=x_t.device))
            eps *= torch.sqrt(1. - self.alcps[t])
            x_0 = (x_t - eps) / self.alcps[t].sqrt()
            x_0.clamp_(-1, 1)

            alpcs_prev = torch.tensor(1., device=x_t.device) if t == 0 else self.alcps[t - 1]
            a = self.alphas[t].sqrt() * (1. - alpcs_prev) * x_t
            b = alpcs_prev.sqrt() * self.betas[t] * x_0
            x_t = (a + b) / (1. - self.alcps[t])
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t += self.betas[t].sqrt() * noise
            print(t, torch.std_mean(x_t), x_t.min(), x_t.max())
        return x_t

device = torch.device('cuda:0')
dm = Diffusion(image_size, time_range, chkpt_file)
dm = dm.to(device)
sample = dm()
sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
sample = sample.permute(0, 2, 3, 1)
sample = sample.contiguous()

import numpy as np
np.savez('test', sample.cpu())
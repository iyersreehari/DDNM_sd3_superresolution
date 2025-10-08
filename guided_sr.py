from math import log, floor, ceil
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from utils.preprocess import PatchifyImage
from tqdm import tqdm

def bicubic_kernel(
        size: int,
        a: float = -0.5
):
    """
    implement 1d Keys bicubic kernel

    :param size:
        (int) kernel size
    :param a:
        (float) Defaults to 0.5
    :return:
        (np.ndarray) bicubic kernel
    """
    x = np.arange(-2, 2, 4 / size) + (2 / size)
    k = np.where(abs(x) <= 1, (a + 2) * (abs(x) ** 3) - (a + 3) * (abs(x) ** 2) + 1,
                 a * (abs(x) ** 3) - 5 * a * (abs(x) ** 2) + 8 * a * (abs(x)) - 4 * a)
    return k / np.sum(k)

class SRConv:
    def __init__(
            self,
            lr_img_size: int,
            channels: int,
            scale: int,
            kernel: torch.Tensor,
            device: str,
            dtype: torch.dtype = torch.bfloat16,
    ):
        """
        SRConv as described in https://github.com/wyhuai/DDNM/blob/main/functions/svd_operators.py

        :param lr_img_size:
            (int) LR image size
        :param channels:
            (int) number of channels
        :param scale:
            (int) sr scale factor
        :param kernel:
            (torch.Tensor) bicubic kernel
        :param device:
            (str)
        :param dtype:
            (torch.dtype)
        """
        hr_img_size = lr_img_size * scale

        A = torch.zeros(lr_img_size, hr_img_size, dtype=torch.float, device=device)

        left = torch.Tensor([((i - 2) * scale + scale // 2) for i in range(lr_img_size)]).to(device)
        right = torch.Tensor([((i + 2) * scale + scale // 2) for i in range(lr_img_size)]).to(device)
        j = torch.vstack([torch.arange(left[i], right[i], dtype=torch.long, device=device)
                          for i in range(lr_img_size)])
        j = torch.where(j >= 0, j, -j - 1)
        j = torch.where(j < hr_img_size, j, 2 * hr_img_size - j - 1)
        # ij = j + (hr_img_size * torch.arange(lr_img_size, dtype=torch.long, device=device)[:, None])

        for k in range(lr_img_size):
            for l in range(j.shape[-1]):
                A[k][j[k][l]] += kernel[l]
        # A_shape = A.shape
        # A = A.flatten()
        # for k in range(ij.shape[0]):
        #     for l in range(ij.shape[1]):
        #         A[ij[k][l]] += kernel[l]
        # A = A.reshape(A_shape)
        # A = A.put_(ij, kernel[idx], accumulate=True)
        # self.A = A
        # A /= A.sum(dim=0)

        self.U, S, self.V = torch.svd(A, some=False)
        self.U = self.U.to(dtype=dtype)
        self.V = self.V.to(dtype=dtype)

        self.kron_S = torch.kron(S, S).unsqueeze(0).repeat(channels, 1)
        self.kron_S_pinv = self.kron_S.clone()
        self.kron_S_pinv[self.kron_S_pinv == 0] = torch.inf
        self.kron_S_pinv = 1 / self.kron_S_pinv

        self.kron_S = self.kron_S.to(dtype=dtype)
        self.kron_S_pinv = self.kron_S_pinv.to(dtype=dtype)

        self.sparse_idx = [hr_img_size * i + j for i in range(lr_img_size) for j in range(lr_img_size)]

        self.device = device
        self.dtype = dtype

    def A(
            self,
            img: torch.Tensor
    ):
        device = img.device
        dtype = img.dtype
        img = img.to(dtype=self.dtype, device=self.device)
        out = (self.V.T @ img) @ self.V
        out = out.flatten(-2)[..., self.sparse_idx]
        out = self.kron_S * out
        out = out.reshape(out.shape[:-1] + self.U.shape)
        out = (self.U @ out) @ self.U.T
        out = out.clip(max=1., min=0.)
        return out.to(dtype=dtype, device=device)

    def A_pinv(
            self,
            img: torch.Tensor
    ):
        device = img.device
        dtype = img.dtype
        img = img.to(dtype=self.dtype, device=self.device)
        out = (self.U.T @ img) @ self.U
        out = out.flatten(-2)
        out = self.kron_S_pinv * out[..., torch.arange(out.shape[-1])]
        temp_out = torch.zeros(out.shape[:-1] + (self.V.shape[0] ** 2,)) \
                    .to(dtype=self.dtype, device=self.device)
        temp_out[..., self.sparse_idx] = out
        out = temp_out.reshape(temp_out.shape[:-1] + self.V.shape)
        out = (self.V @ out) @ self.V.T
        out = out.clip(max=1., min=0.)
        return out.to(dtype=dtype, device=device)


class DDNMsr:
    def __init__(
            self,
            vae,
            latent_model,
            scheduler,
            kernel: torch.Tensor,
            scale: int,
            prompt_embeds: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Adapts DDNM based SR method as described in https://arxiv.org/pdf/2212.00490 for SD3/SD3.5

        :param vae:
            diffusers model for obtaining latent representation
        :param latent_model:
            diffusers model for denoising
        :param scheduler:
            diffusion scheduler
        :param kernel:
            (torch.Tensor) superresolution kernel
        :param scale:
            (int) sr scale factor
        :param prompt_embeds:
            (torch.Tensor) null prompt embeds required latent diffusion model
        :param pooled_prompt_embeds:
            (torch.Tensor) pooled null prompt embeds required latent diffusion model
        :param device:
            (str)
        :param dtype:
            (torch.dtype)
        """
        self.vae = vae
        self.latent_model = latent_model
        self.scheduler = scheduler
        self.kernel = kernel
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=latent_channels
        )

    def vae_encode(
            self,
            x: torch.Tensor,
    ):
        """
        Encode `x` to latent representation, expects x to be in [0., 1.]

        :param x:
            (torch.Tensor) x to encode
        :return:
            (torch.Tensor) latent representation of x
        """
        with torch.no_grad():
            x = self.image_processor.preprocess(x.clamp(min=0., max=1.), height=x.shape[-2], width=x.shape[-1])
            x = self.vae.encode(x).latent_dist.sample()
            x = (x + self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return x

    def vae_decode(
            self,
            x: torch.Tensor,
    ):
        """
        Decode `x` to pixel space, output range - [0., 1.]

        :param x:
            (torch.Tensor) x to decode
        :return:
            (torch.Tensor) pixel space representation of x, range in [0., 1.]
        """
        with torch.no_grad():
            x = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            x = self.vae.decode(x).sample
            x = self.image_processor.postprocess(x, output_type='pt')  # output in [0,1]
        return x

    def refine_x0(
            self,
            x0: torch.Tensor,
            guidance_strength: float = 1.0,
    ):
        """
        x = A_inv * y + guidance_strength * (I - A_inv * A) * x

        :param x0:
            (torch.Tensor)
        :param guidance_strength:
            (float) strength (hyperparam) for guidance
        :return:
            (torch.Tensor)
        """
        x = self.vae_decode(x0)
        g = self.sr_operator.A_pinv(self.sr_operator.A(x))
        x0 = self.A_pinv_y_latent + guidance_strength * (x0 - self.vae_encode(g))
        return x0

    def eps(
            self,
            x_t: torch.Tensor,
            t_i: int,
            guidance_strength: float = 3.5,
    ):
        """
        compute eps with guidance for sr

        :param x_t:
            (torch.Tensor)
        :param t_i:
            (int) current timestep
        :param guidance_strength:
            (float) strength (hyperparam) for guidance
        :return:
            (torch.Tensor)
        """
        t = self.scheduler.timesteps[t_i]
        with torch.no_grad():
            eps = self.latent_model(x_t, timestep=t.unsqueeze(0),
                                    encoder_hidden_states=self.prompt_embeds,
                                    pooled_projections=self.pooled_prompt_embeds).sample
        x0_dt = - self.scheduler.sigmas[t_i]
        if x0_dt == 0:
            return eps
        x0 = x_t + x0_dt * eps
        x0 = self.refine_x0(x0, guidance_strength=guidance_strength)
        eps = (x0 - x_t) / x0_dt
        return eps

    def step(
            self,
            eps: torch.Tensor,
            x_t: torch.Tensor,
            t_i: int,
            stochastic: bool = False,
    ):
        """
        computes x_(t-1) from x_t

        :param eps:
            (torch.Tensor) noise predicted from latent diffusion model
        :param x_t:
            (torch.Tensor)
        :param t_i:
            (int) current timestep
        :param stochastic:
            (bool) if True stochastic else deterministic flow matching prediction
        :return:
        """
        if not stochastic:
            dt = self.scheduler.sigmas[t_i + 1] - self.scheduler.sigmas[t_i]
            prev_sample = x_t + dt * eps
            return prev_sample
        else:
            x0 = x_t - self.scheduler.sigmas[t_i] * eps
            noise = torch.randn_like(x_t)
            prev_sample = (1.0 - self.scheduler.sigmas[t_i + 1]) * x0 + self.scheduler.sigmas[t_i + 1] * noise
            return prev_sample

    def __call__(
            self,
            img: torch.Tensor,
            num_inference_steps: int,
            guidance_strength: float = 3.5,
    ):
        """
        Runs latent space denoising for `num_inference_steps` steps

        :param img:
            (torch.Tensor) LR image
        :param num_inference_steps:
            (int) number of diffusion steps
        :param guidance_strength:
            (float) strength (hyperparam) for guidance
        :return:
            (torch.Tensor) Superresolved image
        """
        device = img.device
        dtype = img.dtype
        img = img.to(dtype=self.dtype, device=self.device)

        self.sr_operator = SRConv(img.shape[-1], img.shape[-3],
                                    self.scale, self.kernel, self.device, self.dtype)

        A_pinv_y = self.sr_operator.A_pinv(img)
        self.A_pinv_y_latent = self.vae_encode(A_pinv_y)

        latent_shape = (img.shape[0],
                        self.latent_model.config.in_channels,
                        self.A_pinv_y_latent.shape[-2],
                        self.A_pinv_y_latent.shape[-1])
        x_t = torch.randn(latent_shape, dtype=self.dtype, device=self.device)
        for t_i in range(num_inference_steps):
            eps = self.eps(x_t, t_i, guidance_strength)
            x_t = self.step(eps, x_t, t_i)
        x_0 = self.vae_decode(x_t)
        return x_0.to(dtype=dtype, device=device)


def diffusion_guided_sr(
        vae,
        latent_model,
        scheduler,
        lr_img: torch.Tensor,
        num_inference_steps: int,
        scale: int,
        guidance_strength: float,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
):
    """
    Helper function to run guided SR

    :param vae:
        diffusers model for obtaining latent representation
    :param latent_model:
        diffusers model for denoising
    :param scheduler:
        diffusion scheduler
    :param lr_img:
        (torch.Tensor) LR image
    :param num_inference_steps:
        (int) number of diffusion steps
    :param scale:
        (int) sr scale factor
    :param guidance_strength:
        (float) strength (hyperparam) for guidance
    :param prompt_embeds:
        (torch.Tensor) null prompt embeds required latent diffusion model
    :param pooled_prompt_embeds:
        (torch.Tensor) pooled null prompt embeds required latent diffusion model
    :param device:
        (str)
    :param dtype:
        (torch.dtype)
    :return:
    """
    out_dtype = lr_img.dtype
    lr_img = lr_img.to(dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    overlap = 2 * scale
    patchify = PatchifyImage(lr_img, overlap=overlap)

    # latent overlap is 2*scale
    # require 2*2*scale + patch_size =  vae.config.sample_size//scale

    lr_img_patches = patchify.pad_and_split_image((vae.config.sample_size // scale) - 2 * overlap)

    kernel = torch.from_numpy(bicubic_kernel(4*scale)).to(dtype=dtype, device=device)

    sr_solver = DDNMsr(vae, latent_model,
                    scheduler, kernel, scale,
                    prompt_embeds, pooled_prompt_embeds,
                    device=device, dtype=dtype)

    hr_img = None
    for patch_idx in tqdm(range(lr_img_patches.shape[0]),
                          desc="Scaling Patches", total=lr_img_patches.shape[0]):
        hr_patch = sr_solver(lr_img_patches[patch_idx].unsqueeze(0), num_inference_steps, guidance_strength)
        if hr_img is None:
            hr_img = hr_patch.detach().cpu()
        else:
            hr_img = torch.cat((hr_img, hr_patch.detach().cpu()), dim=0)
    hr_img = patchify.stitch_and_crop(hr_img, scale)
    return hr_img.to(out_dtype)
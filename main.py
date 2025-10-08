import argparse
import torch
import numpy as np
from torchvision.utils import save_image
import random
from huggingface_hub import login

from guided_sr import diffusion_guided_sr
from utils.diffusion import initialize
from utils.preprocess import load_img


def set_seed(
        seed: int = 0,
        device: str = "cuda"):
    """
    Seed the experiment for reproducibility

    :param seed:
        (int) random seed
    :param device:
        (str) device
    :return:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token",
        type=str,
        help="HuggingFace api token to login"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--lr_img_path",
        type=str,
        help="Path to LR image to be upsampled"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_strength",
        type=float,
        help="Projection strength for ddnm inversion"
    )
    parser.add_argument(
        "--scale",
        type=int,
        help="SR scale"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the experiment"
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="result.png",
        help="Save path for the HR result image"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    login(token=args.hf_token)
    set_seed(int(args.seed), args.device)
    pipe, scheduler, prompt_embeds, pooled_prompt_embeds = \
        initialize(args.model_id, args.device, torch.bfloat16)

    lr_img = load_img(args.lr_img_path)
    final_shape = [lr_img.shape[-1]*args.scale, lr_img.shape[-2]*args.scale]
    hr_img = diffusion_guided_sr(
                pipe.vae,
                pipe.transformer,
                scheduler,
                lr_img,
                int(args.num_inference_steps),
                int(args.scale),
                args.guidance_strength,
                prompt_embeds,
                pooled_prompt_embeds,
                args.device,
                torch.bfloat16,
            )
    hr_img = hr_img[...,:final_shape[0],:final_shape[1]]
    save_image(hr_img, args.savepath)

if __name__ == "__main__":
    main()
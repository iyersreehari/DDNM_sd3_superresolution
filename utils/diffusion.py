import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, \
                        BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler


def load_diffusers_pipeline(
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
):
    """
    Loader for diffusers pipelines.

    :param model_id:
        (str) Hugging Face model id
    :param dtype:
        (torch.dtype) Defaults to torch.bfloat16
    :return:
        initialized diffusers pipeline object and scheduler
    """

    if model_id.startswith("stabilityai/stable-diffusion-3"):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=dtype
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype
        )
    else:
        raise NotImplementedError(f"This loader only supports SD3/SD3.5, got {model_id}")

    pipe.enable_model_cpu_offload()
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe, scheduler

def get_prompt_embeds(
    pipe,
    device: str = 'cuda'
):
    """
    get the null prompt embeds and pooled null prompt embeds

    :param device:
        (str) Defaults to 'cuda'
    :return:
        prompt embeds and pooled prompt embeds needed by SD3/SD3.5 latent model
    """

    # from huggingface.diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = \
            pipe._get_clip_prompt_embeds(
                prompt=[""],
                device=device,
                num_images_per_prompt=1,
                clip_skip=None,
                clip_model_index=0,
            )
        prompt_embeds_2, pooled_prompt_embeds_2 = \
            pipe._get_clip_prompt_embeds(
                prompt=[""],
                device=device,
                num_images_per_prompt=1,
                clip_skip=None,
                clip_model_index=1,
            )
        t5_prompt_embed = pipe._get_t5_prompt_embeds(
            prompt=[""],
            num_images_per_prompt=1,
            max_sequence_length=256,
            device=device,
        )
        clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds

def initialize(
        model_id: str,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
):
    """
    initialize experiment with diffusion model and scheduler

    :param model_id:
        (str) Hugging Face model id
    :param device:
        (str) device to be used.
    :param dtype:
        (torch.dtype) Defaults to torch.bfloat16
    :return:
    """
    pipe, scheduler = load_diffusers_pipeline(model_id, dtype)
    prompt_embeds, pooled_prompt_embeds = get_prompt_embeds(pipe, device)

    return pipe, scheduler, prompt_embeds, pooled_prompt_embeds

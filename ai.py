import os
from distutils.util import strtobool
from io import BytesIO

from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, ShapEPipeline, StableDiffusionXLImg2ImgPipeline, \
    EulerAncestralDiscreteScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Callable, Tuple, Any, Dict, Coroutine
import asyncio
import functools
import typing

from prompts.default_prompt import DefaultPrompt


class AI:

    def to_thread(func: typing.Callable) -> typing.Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(func, *args, **kwargs)

        return wrapper

    @to_thread
    def infer_text(self, prompt, model="TinyLlama-1.1B-Chat-v1.0"):
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model}")
        model = AutoModelForCausalLM.from_pretrained(f"models/{model}")
        inputs = tokenizer(DefaultPrompt().build(prompt), return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return DefaultPrompt().clean(prompt, response)

    @to_thread
    def infer_text_image(self, prompt, model="stable-diffusion-xl-base-1.0", refine=True):

        pipe = DiffusionPipeline.from_pretrained(f"models/{model}", torch_dtype=torch.float16,
                                                 use_safetensors=True, variant="fp16")
        if strtobool(os.getenv('IS_GPU', 'no')):
            pipe.to("cuda")

        result = pipe(prompt=prompt)
        image = BytesIO()
        img = result.images[0]
        img.save(image, format="PNG")
        image.seek(0)
        if refine:
            image = self.image_refiner(prompt, image)
        return image

    def image_refiner(self, prompt, image, model='stable-diffusion-xl-refiner-1.0'):
        init_image = Image.open(image)
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(f"models/{model}", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, safety_checker=None)
        if strtobool(os.getenv('IS_GPU', 'no')):
            pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        result = pipe(prompt=prompt, image=init_image, num_inference_steps=10, image_guidance_scale=1)
        image = BytesIO()
        img = result.images[0]
        img.save(image, format="PNG")
        image.seek(0)
        return image

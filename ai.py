import os
from io import BytesIO
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, ShapEPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    def infer_text(self, prompt, model="TinyLlama-1.1B-Chat-v1.0"):
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model}")
        model = AutoModelForCausalLM.from_pretrained(f"models/{model}")
        inputs = tokenizer(DefaultPrompt().build(prompt), return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=20)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    @to_thread
    def infer_text_image(self, prompt, model="stable-diffusion-xl-base-1.0"):

        if os.getenv('IS_GPU', 'No') == 'Yes':
            pipe = DiffusionPipeline.from_pretrained(f"models/{model}", torch_dtype=torch.float16,
                                                     use_safetensors=True, variant="fp16")
            pipe.to("cuda")
            result = pipe(prompt=prompt)
        else:
            pipe = DiffusionPipeline.from_pretrained(f"models/{model}", torch_dtype=torch.float32,
                                                     use_safetensors=True, variant="fp16")
            result = pipe(prompt=prompt)

        image = BytesIO()
        img = result.images[0]
        img.save(image, format="PNG")
        image.seek(0)
        return image

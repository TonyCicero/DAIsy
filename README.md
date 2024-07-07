# DAIsy

## Description:
DAIsy is a Discord bot with AI Integrations

## Setup:
**Environment Variables**:
```
DISCORD_TOKEN=XXXXXX
IS_GPU=<yes/no>
REFINE_IMAGE=<yes/no>
```

Install Requirements (Conda):
```commandline
conda install -c conda-forge diffusers
pip install git+https://github.com/huggingface/transformers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers[sentencepiece]
pip install optimum
pip install auto-gptq
```

Recommended Base Models
- Models should be cloned into a '/models' directory in the root of this repository

Chat: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

Image Generation: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

Image Refinement: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

**GPU Support**

For faster inferences It is recommended to run on a device with a compatible GPU.
- PyTorch version must be compiled with cuda support 
- ENV variable `IS_GPU` must be set to `yes`

**Image Refinement**

Generated images can be run through a refiner model to remove image artifacts and provide an overall better quality image.
- Image refinement adds additional delay to image generation
- To enable, ENV variable `REFINE_IMAGE` must be set to `yes`

## Usage:
Discord Commands:
```text
!chat <text generation prompt>
!image <image generation prompt>
```

# DAIsy

## Description:
DAIsy is a Discord bot with AI Integrations

## Setup:
**Environment Variables**:
```
DISCORD_TOKEN=XXXXXX
IS_GPU=<Yes/No>
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

https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

**GPU Support**

For faster inferences It is recommended to run on a device with a compatible GPU.
- PyTorch version must be compiled with cuda support 
- ENV variable `IS_GPU` must be set to `Yes`

## Usage:

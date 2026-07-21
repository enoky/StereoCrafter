# 🛠️ Manual Installation Guide for StereoCrafter  for Windows (UV Edition)

This guide walks you through manually installing the **StereoCrafter** using the uv package manager.

---

## 📋 Prerequisites

Ensure the following tools are installed and available in your system's PATH:

- [Git](https://git-scm.com/)
- A recent [NVIDIA GPU driver](https://www.nvidia.com/drivers/) (CUDA 13 capable, version 580 or newer)
- [FFMPEG](https://techtactician.com/how-to-install-ffmpeg-and-add-it-to-path-on-windows/)

> The CUDA Toolkit is **not** required. All CUDA components ship prebuilt with the Python packages: PyTorch cu130 wheels bundle the CUDA runtime, SageAttention installs as a prebuilt wheel, and triton-windows bundles its own compiler tools.

---

## 🚀 Installation Steps

### 1. Verify NVIDIA Driver

Check that your driver supports CUDA 13:

```bash

nvidia-smi

```

Look for `CUDA Version: 13.0` (or higher) in the top-right corner of the output.

> If the reported CUDA version is lower than 13.0, update your driver from [NVIDIA's driver page](https://www.nvidia.com/drivers/).

### 2. Install uv Package Manager

```bash

powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

```


### 3. Clone the Repository with Submodules & Change Directory Path

```bash

git clone --recursive https://github.com/enoky/StereoCrafter.git
cd StereoCrafter

```

> If the folder `StereoCrafter` already exists, delete or rename it before proceeding.



---

### 4. Setup and Install

```bash

uv sync

```


## 📦 Model Weights Installation

The models are required to run StereoCrafter. You can download them automatically using the Hugging Face CLI (provided by the huggingface-hub package in your environment).

### 1. Prerequisite: Accept Model Terms

The SVD model is "Gated." You must manually accept the terms of use before you can download it:

- 1. Log in to [Hugging Face.](https://huggingface.co/)
- 2. Visit the [SVD XT 1.1 Page.](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
- 3. Click **"Agree and access repository"**.

### 2. Authenticate your PC
You need a "Read" Token from your [Hugging Face Settings.](https://huggingface.co/settings/tokens) Run this command and paste your token when prompted:

```bash

uv run hf auth login

```

### 3. Download the Models
Run these commands from the `StereoCrafter` root folder. The CLI will automatically create the `weights` folder if it doesn't exist:

```bash
# 1. Download SVD img2vid (Gated - Requires Login)
uv run hf download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir weights/stable-video-diffusion-img2vid-xt-1-1

# 2. Download DepthCrafter (Open)
uv run hf download tencent/DepthCrafter --local-dir weights/DepthCrafter

# 3. Download StereoCrafter (Open)
uv run hf download TencentARC/StereoCrafter --local-dir weights/StereoCrafter

```

**Note:** Total download size is approximately 22GB. Ensure you have enough disk space.
Your downloaded weights should contain files matching the visual hiearchy below.

```text

..\Stereocrafter\weights\
   │ 
   ├───DepthCrafter/
   │       config.json                                  
   │       diffusion_pytorch_model.safetensors          
   │                                                    
   ├───stable-video-diffusion-img2vid-xt-1-1/  
   │   │   model_index.json                            
   │   │   svd_xt_1_1.safetensors                       
   │   │                                                
   │   ├───feature_extractor/
   │   │       preprocessor_config.json                 
   │   │                                                
   │   ├───image_encoder/
   │   │       config.json                              
   │   │       model.fp16.safetensors                   
   │   │       model.safetensors                        
   │   │                                                
   │   ├───scheduler/
   │   │       scheduler_config.json                    
   │   │                                                
   │   ├───unet/
   │   │       config.json                              
   │   │       diffusion_pytorch_model.fp16.safetensors 
   │   │       diffusion_pytorch_model.safetensors      
   │   │                                                
   │   └───vae/
   │           config.json                              
   │           diffusion_pytorch_model.fp16.safetensors 
   │           diffusion_pytorch_model.safetensors      
   │                                                    
   └───StereoCrafter/                        
           config.json                                  
           diffusion_pytorch_model.safetensors         

```

## ✅ Final Notes

- If any step fails, check your environment variables and permissions.
- Refer to `install_log.txt` (if generated during script-based install) for troubleshooting.
- CUDA support is critical for GPU acceleration. Ensure your NVIDIA driver is up to date (CUDA 13 capable).


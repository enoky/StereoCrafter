# 🛠️ Manual Installation Guide for StereoCrafter  for Windows (UV Edition)

This guide walks you through manually installing the **StereoCrafter** using the uv package manager.

---

## 📋 Prerequisites

Ensure the following tools are installed and available in your system's PATH:

- [Git](https://git-scm.com/)
- [CUDA Toolkit 12.8 or 12.9](https://developer.nvidia.com/cuda-toolkit)
- [FFMPEG](https://techtactician.com/how-to-install-ffmpeg-and-add-it-to-path-on-windows/)

---

## 🚀 Installation Steps

### 1. Verify CUDA Toolkit Installation

Check that `nvcc` is available and the version is 12.8 or 12.9:

```bash

nvcc --version

```

Look for output like:

```

Cuda compilation tools, release 12.8, V12.8.89

```

> If `nvcc` is not found or the version is incorrect, install the correct version from [NVIDIA's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit).

---

### 2. Install uv Package Manager

```bash

powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

```

---

### 3. Clone the Repository with Submodules & change directory

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

---

## Stereocrafter weights install

Download and extract [model](https://mega.nz/file/Fw1GgJrL#bPplu2Y1PT4G-TM29zcGNENUYVySEk2NENT4krkjEso) "weights" to StereoCrafter folder (use <a href="https://www.qbittorrent.org">qBittorrent</a> to download)

Or manually download from the original locations below..


#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd weights
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download the [DepthCrafter model](https://huggingface.co/tencent/DepthCrafter) for the video depth estimation.
```bash

git clone https://huggingface.co/tencent/DepthCrafter

```

#### 3. Download the [StereoCrafter model](https://huggingface.co/TencentARC/StereoCrafter) for the stereo video generation.
```bash

git clone https://huggingface.co/TencentARC/StereoCrafter

```

You downloaded weights files should match the visual hiearchy below

```text

..\Stereocrafter\weights\
   │ 
   ├───DepthCrafter/
   │       config.json                                  
   │       diffusion_pytorch_model.safetensors          
   │                                                    
   ├───stable-video-diffusion-img2vid-xt-1-1/
   │   │   gitattributes                                
   │   │   LICENSE.md                                   
   │   │   model_index.json                             
   │   │   README.md                                    
   │   │   svd11.webp                                   
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
           .gitattributes                               
           config.json                                  
           diffusion_pytorch_model.safetensors          
           LICENSE                                      
           NOTICE                                       
           README.md

```
---

```

> If this fails, ensure your NVIDIA driver, CUDA Toolkit, and PyTorch installation are compatible.

---

## ✅ Final Notes

- If any step fails, check your environment variables and permissions.
- Refer to `install_log.txt` (if generated during script-based install) for troubleshooting.
- CUDA support is critical for GPU acceleration. Ensure your drivers and toolkit are correctly installed.


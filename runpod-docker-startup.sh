#!/bin/bash
set -e

echo "=========================================="
echo "StereoCrafter WEBUI - Quick Start"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""

echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set!"
    echo "Please set HF_TOKEN in RunPod environment variables"
    exit 1
fi

# Create weights folder if it doesn't exist
mkdir -p weights
cd weights

# Check if models already exist (for persistent storage)
if [ -d "stable-video-diffusion-img2vid-xt-1-1" ] && \
   [ -d "DepthCrafter" ] && \
   [ -d "StereoCrafter" ]; then
    echo "âœ… All model weights already downloaded"
else
    echo "Downloading model weights..."
    
    # Configure git credential helper
    git config --global credential.helper store
    
    # Login to HuggingFace using Python (this properly sets up credentials)
    echo "Logging into Hugging Face..."
    python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'].strip(), add_to_git_credential=True)"
    
    # Verify login worked
    if [ $? -ne 0 ]; then
        echo "ERROR: HuggingFace login failed"
        exit 1
    fi
    
    echo "HuggingFace login successful"
    
    # Download stable-video-diffusion using huggingface-cli (with Python fallback)
    if [ ! -d "stable-video-diffusion-img2vid-xt-1-1" ]; then
        echo "Downloading stable-video-diffusion-img2vid-xt-1-1..."
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir stable-video-diffusion-img2vid-xt-1-1
        else
            echo "huggingface-cli not found, using Python fallback..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-video-diffusion-img2vid-xt-1-1', local_dir='stable-video-diffusion-img2vid-xt-1-1', local_dir_use_symlinks=False)"
        fi
    fi
    
    # Download DepthCrafter
    if [ ! -d "DepthCrafter" ]; then
        echo "Downloading DepthCrafter..."
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download tencent/DepthCrafter --local-dir DepthCrafter
        else
            python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/DepthCrafter', local_dir='DepthCrafter', local_dir_use_symlinks=False)"
        fi
    fi
    
    # Download StereoCrafter weights
    if [ ! -d "StereoCrafter" ]; then
        echo "Downloading StereoCrafter weights..."
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download TencentARC/StereoCrafter --local-dir StereoCrafter
        else
            python -c "from huggingface_hub import snapshot_download; snapshot_download('TencentARC/StereoCrafter', local_dir='StereoCrafter', local_dir_use_symlinks=False)"
        fi
    fi
    
    echo "âœ… All models downloaded"
fi

cd ..

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Starting StereoCrafter WEBUI..."
echo ""

# Start the application
python webui.py --share --server-name 0.0.0.0 --server-port 7860

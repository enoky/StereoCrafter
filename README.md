# Contributing to StereoCrafter WEBUI

## 🎯 About This Project

This is a **web-based UI fork** of the original [StereoCrafter](https://github.com/TencentARC/StereoCrafter) project by Tencent ARC Lab and by [Enoky](https://github.com/enoky/StereoCrafter) for the GUI implementation. This version is tested with a minimum RTX 6000 Ada ( Lovelace architecture ) and 48GB VRAM.

### Why a Web UI?

The original StereoCrafter uses CLI and Enoky tkinter-based desktop GUIs, which work great locally but have limitations:

- **Limited to local hardware** - You're stuck with the GPU you bought. You cannot increase the VRAM.
- **No remote access** - You can't process videos from anywhere.
- **Resource intensive** - High-end GPUs are expensive to own. Yet they output results faster.
- **OS, Batch and UI** - StereoCrafter is command-line usage only and can't do batch video file processing. Enoky is GUI-based and batch-processing enabled, but the python code is optimized for Windows 10/11 and might generate errors running on Linux/MacOS.

### Our Solution: Cloud-Ready Web Interface

By converting to a **Gradio-based web UI**, we enabled:

✅ **RunPod Integration** - Access high-end GPUs (RTX 4090, A6000) for pennies per hour  
✅ **Remote Processing** - Process videos from any device with a browser  
✅ **Cost Effective** - Pay only for GPU time you use, not ownership  
✅ **Scalability** - Spin up multiple instances for batch processing  
✅ **Accessibility** - Share processing power with team members  

**Example Cost Savings:**
- RTX 4090 ownership: ~$1,600 upfront
- RunPod RTX 4090: ~$0.50/hour (only when processing)
- Process 10 hours/month: $5 vs $1,600 investment

### Honorable Mention

All credit for the core StereoCrafter technology goes to the original authors at **Tencent ARC Lab**. This project simply wraps their excellent work in a web interface for cloud deployment.

Original Paper: [StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos](https://arxiv.org/abs/2409.07447)

### Project Status

⚠️ **This is a passion project built in free time!**

This codebase is:
- **Functional but not optimized** - It works, but there's room for improvement
- **Built with enthusiasm** - Created out of love for the technology, not commercial purposes
- **Community-driven** - Your contributions can help make it better!

We welcome optimizations, bug fixes, and feature improvements. This is a learning project for everyone involved!

---

# Quick Runpod deploy
## 🚀 Runpod Setup Guide

### 1. Select Pod Template
*   Navigate to **Pod Template**
*   Select **All**
*   Choose **Runpod Pytorch 2.8.0**

### 2. Configure Hardware
*   Click **Configure Pod**
*   Select GPU: **RTX 6000 Ada**
NOTE: Do not select a GPU that is below 48GB VRAM or GPUs that are not using Lovelace architecture. RTX 6000 Ada is the minimum GPU you should run this version of StereoCrafter.

### 3. Edit Container Settings
Under **Pod Template**, click **Edit** and enter the following:

| Setting | Value |
| :--- | :--- |
| **Container Name** | `johnsdoes/stereocrafter-webui:latest` |
| **Container Disk** | `300 GB` *(Cost: ~$0.042/hr)* |
| **Volume Disk** | `0 GB` |
| **Expose HTTP Ports Stereocrafter** | `7860` |
| **Expose HTTP Ports FileBrowser** | `8080` |

### 4. Environment Variables
*   **Key:** `HF_TOKEN`
*   **Value:** `your huggingface read token`
*   **Key:** `FB_USERNAME`
*   **Value:** `your Filebrowsr username`
*   **Key:** `FB_PASSWORD`
*   **Value:** `your Filebrowser password`



### 5. Deploy
*   Click **Set Overrides**
*   Select **Deploy On-Demand**

## 🚀 Local Setup 

### Option 1: Docker (Recommended - Fastest & Easiest)

**Why Docker?**
- ✅ No dependency conflicts or DLL issues (especially on Windows)
- ✅ Simulates RunPod deployment environment exactly
- ✅ Pre-configured with all dependencies
- ✅ Works on Windows, Linux, and Mac
- ✅ One command to start everything

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- NVIDIA GPU with drivers installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

**Quick Start:**

```bash
# Clone the repository
git clone https://github.com/keemzin/StereoCrafter.git
cd StereoCrafter

# Create .env file with your HuggingFace token
echo HF_TOKEN=your_huggingface_token_here > .env

# Start the container (downloads models on first run)
docker compose up -d

# View logs
docker compose logs -f

# Access the WebUI at http://localhost:7860
```

**With FileBrowser (optional):**

FileBrowser gives you a web-based file manager at port 8080 to browse/upload/download your input and output files.

Set credentials in `.env` before first run (minimum 12 characters for password):

```env
HF_TOKEN=your_huggingface_token_here
FB_USERNAME=admin
FB_PASSWORD=adminadmin12
```

Then start with the FileBrowser compose file:

```bash
# Build the FileBrowser image (first time only)
docker build -f Dockerfile.filebrowser -t stereocrafter-filebrowser:local .

# Start with FileBrowser
docker-compose -f docker-compose.filebrowser.yml up -d

# Access:
#   StereoCrafter WebUI → http://localhost:7860
#   FileBrowser         → http://localhost:8080
```

To reset FileBrowser credentials:
```bash
docker-compose -f docker-compose.filebrowser.yml down
docker volume rm stereocrafter_filebrowser-data
docker-compose -f docker-compose.filebrowser.yml up -d
```

**Customizing Mounts:**

Edit `docker-compose.yml` to mount additional folders:

```yaml
volumes:
  # Your code (for live development)
  - ./webui.py:/workspace/StereoCrafter/webui.py
  - ./gui:/workspace/StereoCrafter/gui
  - ./stereocrafter_ui:/workspace/StereoCrafter/stereocrafter_ui
  
  # Model weights (persistent Docker volume - no Windows path issues)
  - stereocrafter-weights:/workspace/StereoCrafter/weights
  
  # Input/output folders (modify as needed)
  - ./source_video:/workspace/StereoCrafter/source_video
  - ./output_depthmaps:/workspace/StereoCrafter/output_depthmaps
  - ./output_splatted:/workspace/StereoCrafter/output_splatted
  - ./completed_output:/workspace/StereoCrafter/completed_output
  - ./final_videos:/workspace/StereoCrafter/final_videos
```

**Docker Commands:**

```bash
# Stop the container
docker compose down

# Restart after code changes
docker compose restart

# View container status
docker compose ps

# Access container shell
docker compose exec stereocrafter-webui bash

# Remove everything (including downloaded models)
docker compose down -v
```

---

### Option 2: Manual Setup (Python Virtual Environment)

**⚠️ Known Issues on Windows:**
- PyTorch DLL loading errors are common on Windows (`WinError 1114: DLL initialization failed`)
- Requires Visual C++ Redistributables and proper CUDA installation
- Can have conflicts with other Python installations or Anaconda
- **Docker is strongly recommended for Windows users**

**If you still want to try manual setup:**
### 📋 Prerequisites

Before you begin, ensure you have:

- **Git**: [Download here](https://git-scm.com/downloads)
- **Python 3.12**: [Download here](https://www.python.org/downloads)
- **CUDA 12.8**: [Download here](https://developer.nvidia.com/cuda-12-8-0-download-archive) (for GPU support)
- **FFmpeg**: [Installation guide](https://techtactician.com/how-to-install-ffmpeg-and-add-it-to-path-on-windows/)
- **HuggingFace Account**: [Sign up here](https://huggingface.co/join) (HF_TOKEN needed for model downloads)

Verify installations:
```bash
git --version
python --version
ffmpeg -version
```

### Step 1: Clone My StereoCrafter Fork

```bash
# Clone YOUR fork (keemzin/StereoCrafter)
git clone https://github.com/keemzin/StereoCrafter.git
cd StereoCrafter

# Add upstream remote for syncing with original enoky/StereoCrafter
git remote add upstream https://github.com/enoky/StereoCrafter.git

# Verify remotes
git remote -v
```

**Why clone from my fork?**

This WEBUI fork includes all the necessary code patches pre-applied. Cloning from `keemzin/StereoCrafter` gives you everything in one step.

**Can I update from the original enoky/StereoCrafter?**

Yes! Use the `upstream` remote to fetch updates from the original repository:

```bash
# Fetch latest from upstream
git fetch upstream

# Check what's new
git log --oneline upstream/main -n 10

# Merge upstream changes (if compatible)
git merge upstream/main
```

### Step 2: Download Model Weights

Create a `weights` folder and download the required models:

```bash
# Create weights folder
mkdir weights
cd weights

# Download models using git
git clone https://huggingface.co/TencentARC/StereoCrafter
git clone https://huggingface.co/tencent/DepthCrafter
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1

cd ..
```

### Step 3: Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

### Step 5: Run the WEBUI

```bash
# Start the web interface
python webui.py
```

The WEBUI will open in your browser at `http://localhost:7860`

**Troubleshooting Windows DLL Errors:**

If you get `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`:

1. **Install Visual C++ Redistributables:**
   - Download: [VC++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Install and restart your computer

2. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **Reinstall PyTorch:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

4. **If still failing, use Docker instead** (recommended solution)

**Why Docker is better for Windows:**
- No DLL conflicts
- Pre-configured environment
- Matches production RunPod environment exactly
- One command to start: `docker compose up -d`

## 🔧 Making Changes

### Project Structure

Only modify files in these locations
- `webui.py` - Main application entry point
- `gui/` - GUI components (legacy, being phased out)
- `stereocrafter_ui/` - New WEBUI components (Gradio-based)

**Do NOT modify:**
- `pipelines/` - Core processing pipelines
- `core/` - Core functionality
- `dependency/` - External dependencies
- Other original enoky/StereoCrafter files

### Development Workflow

1. **Create a new branch for your feature:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** in the allowed folders:
   - `webui.py`
   - `gui/`
   - `stereocrafter_ui/`

3. **Test your changes:**
   ```bash
   python webui.py
   ```

4. **Check for errors:**
   - Test all affected features
   - Verify no console errors
   - Test on different video resolutions if applicable

## 📤 Submitting Your Contribution

### Step 1: Clone/modify/Commit Your Changes

```bash
# Check what files you've modified
git status

# Add your changes
git add webui.py
git add stereocrafter_ui/
git add gui/

# Commit with a descriptive message
git commit -m "Add feature: Brief description of your changes"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issue numbers if applicable

Examples:
```
Add manual preview selector to splatting UI
Fix inpainting pipeline loading for local weights
Update file manager with pipeline folder shortcuts
```

### Step 2: Push to Your Fork

```bash
# Push your branch to your fork
git push myfork feature/your-feature-name
```

### Step 3: Create a Pull Request

1. Go to https://github.com/keemzin/WEBUI-StereoCrafter
2. Click "New Pull Request"
3. Select your branch from your fork
4. Fill in the PR template:

```markdown
## Description
Brief description of what this PR does

## Changes Made
- List of specific changes
- Another change
- etc.

## Testing
How you tested these changes

## Screenshots (if applicable)
Add screenshots showing the changes

## Checklist
- [ ] Code follows project structure (only modified webui.py, gui/, stereocrafter_ui/)
- [ ] Tested locally and works as expected
- [ ] No console errors
- [ ] Updated documentation if needed
```

5. Click "Create Pull Request"

## 🐛 Reporting Issues

If you find a bug but don't know how to fix it:

1. Go to the Issues page
2. Click "New Issue"
3. Provide:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots/logs if applicable
   - Your system info (OS, Python version, GPU)

## 💡 Tips for Contributors

### Code Style
- Follow existing code style in the project
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

### Testing
- Test with different video resolutions
- Test with different parameter combinations
- Check memory usage for large videos
- Verify UI responsiveness

### Documentation
- Update docstrings for new functions
- Add inline comments for complex code
- Update README if adding new features
- Include usage examples

### Common Issues

**PyTorch DLL errors on Windows (`WinError 1114`):**
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll"
```

**Solution:** Use Docker instead. This is a known Windows + PyTorch issue that's difficult to resolve. Docker works perfectly and matches the RunPod environment.

If you must use venv:
1. Install [VC++ Redistributables](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Ensure CUDA 12.8 is properly installed
3. Reinstall PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

**Import errors:**
```bash
# Make sure you're in the virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall requirements
pip install -r requirements.txt
```

**CUDA errors:**
- Verify CUDA 12.8 is installed
- Check GPU drivers are up to date
- Ensure PyTorch detects your GPU: `python -c "import torch; print(torch.cuda.is_available())"`

**Model download fails:**
- Check your HuggingFace token is valid
- Ensure you have accepted the model licenses on HuggingFace
- Try using Python download method instead of `uv`

## 📞 Getting Help

- **Questions:** Open a discussion on the repository
- **Bugs:** Open an issue with details
- **Feature requests:** Open an issue with the "enhancement" label

## 🙏 Thank You!

Your contributions help make StereoCrafter WEBUI better for everyone. We appreciate your time and effort!

---

If you’d like to support this project, please consider registering for RunPod using my referral link:
https://runpod.io?ref=5r6ik1wp

Your support helps me troubleshoot issues, test new features, and continue developing the project. Currently, I’m relying mostly on free credits for testing, so every bit of support helps keep the project alive. 🚀


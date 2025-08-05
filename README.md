# StereoCrafter GUI + DepthCrafter GUI Seg

You can learn more about DepthCrafter GUI Seg <a href="https://github.com/Billynom8/DepthCrafter_GUI_Seg">here</a>.

## Installation (1-click installer script)

#### 1. PREREQUISITES:
   - GIT: Ensure Git is installed and added to your system’s PATH.<br>
     Download here: https://git-scm.com/downloads/win<br>
     You can check the installation by running the command:<br>
       `git --version`<br>
     If it shows a version, Git is installed and on PATH.
   
   - PYTHON: Ensure Python 3.12 is installed and added to your PATH.<br>
     Download here: https://www.python.org/downloads<br>
     You can check by running the command:<br>
       `python --version`
   - CUDA ToolKit: Ensure CUDA 12.8 is installed and added to your PATH.<br>
     Download here: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64<br>

#### 2. INSTALL:
   - Run <a href="https://github.com/enoky/StereoCrafter/blob/90257ee4f032b649530281b0eceb95ea460db115/1click_installer_script/StereoCrafter_1click_Installer.bat">script</a> from folder where you want StereoCrafter installed
   - Download and extract <a href="https://drive.google.com/file/d/1bKL7Hag8AG4WTEX3jFGJA69nO8_kZQLP/view?usp=drive_link">model</a> "weights" to StereoCrafter folder

#### 3. LAUNCH GUI SCRIPTS:
   - First activate virtual environment:
       `venv\scripts\activate.bat`
   - Run DepthCrafter GUI Seg:
       `python dc_gui_seg.py`
   - Run DepthSplat GUI:
       `python depth_splatting_gui.py`
   - Run Inpaint GUI:
       `python inpainting_gui.py`

<hr>
<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author 

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>
</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

![teaser](assets/teaser.jpg)

## 📣 News
- `2024/12/27` We released our inference code and model weights.
- `2024/09/11` We submitted our technical report on arXiv and released our project page.

## 🎞️ Showcases
Here we show some examples of input videos and their corresponding stereo outputs in Anaglyph 3D format.
<div align="center">
    <img src="assets/demo.gif">
</div>


## 🛠️ Installation

#### 1. Set up the environment
We run our code on Python 3.8 and Cuda 11.8.
You can use Anaconda or Docker to build this basic environment.

#### 2. Clone the repo
```bash
# use --recursive to clone the dependent submodules
git clone --recursive https://github.com/TencentARC/StereoCrafter
cd StereoCrafter
```

#### 3. Install the requirements
```bash
pip install -r requirements.txt
```


#### 4. Install customized 'Forward-Warp' package for forward splatting
```
cd ./dependency/Forward-Warp
chmod a+x install.sh
./install.sh
```


## 📦 Model Weights

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd ./weights
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


## 🔄 Inference

Script:

```bash
# in StereoCrafter project root directory
sh run_inference.sh
```

There are two main steps in this script for generating stereo video.

#### 1. Depth-Based Video Splatting Using the Video Depth from DepthCrafter
Execute the following command:
```bash
python depth_splatting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                                    --input_video_path [PATH] --output_video_path [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the DepthCrafter model weights (e.g., `./weights/DepthCrafter`).
- `--input_video_path`: Path to the input video (e.g., `./source_video/camel.mp4`).
- `--output_video_path`: Path to the output video (e.g., `./outputs/camel_splatting_results.mp4`).
- `--max_disp`: Parameter controlling the maximum disparity between the generated right video and the input left video. Default value is `20` pixels.

The first step generates a video grid with input video, visualized depth map, occlusion mask, and splatting right video, as shown below:

<img src="assets/camel_splatting_results.jpg" alt="camel_splatting_results" width="800"/> 

#### 2. Stereo Video Inpainting of the Splatting Video
Execute the following command:
```bash
python inpainting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                               --input_video_path [PATH] --save_dir [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the StereoCrafter model weights (e.g., `./weights/StereoCrafter`).
- `--input_video_path`: Path to the splatting video result generated by the first stage (e.g., `./outputs/camel_splatting_results.mp4`).
- `--save_dir`: Directory for the output stereo video (e.g., `./outputs`).
- `--tile_num`: The number of tiles in width and height dimensions for tiled processing, which allows for handling high resolution input without requiring more GPU memory. The default value is `1` (1 $\times$ 1 tile). For input videos with a resolution of 2K or higher, you could use more tiles to avoid running out of memory.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format, as shown below:

<img src="assets/camel_sbs.jpg" alt="camel_sbs" width="800"/> 

<img src="assets/camel_anaglyph.jpg" alt="camel_anaglyph" width="400"/>

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter): A novel method to generate temporally consistent depth sequences from videos.


## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```

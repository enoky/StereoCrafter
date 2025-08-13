# 🛠️ Manual Installation Guide for StereoCrafter

This guide walks you through manually installing the **StereoCrafter** environment without using the batch script.

---

## 📋 Prerequisites

Ensure the following tools are installed and available in your system's PATH:

- [Git](https://git-scm.com/)
- [Python 3.12](https://www.python.org/)
- [CUDA Toolkit 12.8 or 12.9](https://developer.nvidia.com/cuda-toolkit)

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

### 2. Verify Python Installation

Check Python version:

```bash
python --version
```

> Ensure the version is **exactly 3.12**. Other versions are not supported.

---

### 3. Clone the Repository with Submodules

```bash
git clone --recursive https://github.com/enoky/StereoCrafter.git
```

> If the folder `StereoCrafter` already exists, delete or rename it before proceeding.

---

### 4. Navigate to the Project Directory

```bash
cd StereoCrafter
```

---

### 5. Check for `requirements.txt`

Ensure the file exists:

```bash
dir requirements.txt
```

---

### 6. Create a Virtual Environment

```bash
python -m venv venv
```

---

### 7. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

---

### 8. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### 9. Install Dependencies

```bash
python -m pip install --upgrade -r requirements.txt
```

---



```

> If this fails, ensure your NVIDIA driver, CUDA Toolkit, and PyTorch installation are compatible.

---

## ✅ Final Notes

- If any step fails, check your environment variables and permissions.
- Refer to `install_log.txt` (if generated during script-based install) for troubleshooting.
- CUDA support is critical for GPU acceleration. Ensure your drivers and toolkit are correctly installed.


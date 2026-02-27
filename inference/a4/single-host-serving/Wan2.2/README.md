# Single host inference & Multi host inference benchmark of Wan-AI/Wan2.2-T2V-A14B & Wan-AI/Wan2.2-I2V-A14B with Sglang on A4

This document outlines the steps to serve and benchmark various Large Language Models (LLMs) using the [SGLang](https://github.com/sgl-project/sglang/tree/main) framework
## Before you begin

### 1. Create a GCP VM with A4 GPUs

First, we will create a Google Cloud Platform (GCP) Virtual Machine (VM) that has the necessary GPU resources.

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   You have a project with a GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance. The `MACHINE_TYPE` is set to `g4-standard-384` for a multi-GPU VM (8 GPUs). The boot disk is set to 200GB to accommodate the models and dependencies.

```bash
export VM_NAME="${USER}-a4-sglang-wan2.2"
export PROJECT_ID="your-project-id"
export ZONE="your-zone"
export MACHINE_TYPE="a4-highgpu-8g"
export IMAGE_PROJECT="ubuntu-os-accelerator-images"
export IMAGE_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-570"

gcloud compute instances create ${VM_NAME} \
  --machine-type=${MACHINE_TYPE} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --image-project=${IMAGE_PROJECT} \
  --image-family=${IMAGE_FAMILY} \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

### 2. Connect to the VM

Use `gcloud compute ssh` to connect to the newly created instance.

```bash
gcloud compute ssh ${VM_NAME?} --project=${PROJECT_ID?} --zone=${ZONE?}
```

```bash
# Run NVIDIA smi to verify the driver installation and see the available GPUs.
nvidia-smi
```

## Serve a model

### 1. Settingup Sglang & Run Docker File

```bash
sudo apt-get update
sudo apt-get -y install git git-lfs
mkdir -p ~/sglang_build && cd ~/sglang_build
git clone https://github.com/sgl-project/sglang.git 
cd sglang

# Use CUDA 12.8 native image for Blackwell support
FROM nvcr.io/nvidia/pytorch:25.01-py3
WORKDIR /sgl-workspace

# Build fixes for setuptools_scm
COPY python/pyproject.toml python/
RUN mkdir -p python/sglang && echo "# Placeholder" > python/README.md

# Install dependencies using 'uv'
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv --python 3.12 --seed /opt/venv && \
    source /opt/venv/bin/activate && \
    /root/.local/bin/uv pip install --no-cache-dir nvitop huggingface_hub[cli] accelerate && \
    /root/.local/bin/uv pip install --no-cache-dir --prerelease=allow './python[diffusion]'
COPY . .
RUN source /opt/venv/bin/activate && \
    /root/.local/bin/uv pip install --reinstall --no-cache-dir --no-deps './python[diffusion]'

RUN echo 'source /opt/venv/bin/activate' >> /root/.bashrc

# Build the Docker image
docker build -t sglang-wan-blackwell -f Dockerfile.sgl-wan .

# Run the Docker container
mkdir -p /scratch/cache
docker run --gpus '"device=0"' -it --rm \
    --ipc=host --network=host \
    -v $(pwd):/sgl-workspace/sglang \
    -v /home/pkesana_google_com/Wan2.2-weights:/sgl-workspace/weights \
    sglang-wan-blackwell
```

Now you are inside the container.

### 4. Download model

```bash
# Inside the container
export TMPDIR=/tmp
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir ./weights/T2V
```

## Run Benchmarks

Create a script to run the benchmarks with different configurations.

```bash
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."     --save-output --num-gpus 1 --num-frames 81
## Clean up

### 1. Exit the container

```bash
exit
```

### 2. Delete the VM

This command will delete the GCE instance and all its disks.

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```
.

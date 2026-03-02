# Benchmark of Wan-AI/Wan2.2-T2V-A14B & Wan-AI/Wan2.2-I2V-A14B with Sglang on A4X

This document outlines the steps to serve and benchmark various Large Language Models (LLMs) using the [SGLang](https://github.com/sgl-project/sglang/tree/main) framework
 
### 1.1. Environment setup

### 1.1. Clone the Repository

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(pwd)
export RECIPE_ROOT=$REPO_ROOT/inference/a4/single-host-serving/sglang/Wan2.2
```

<a name="configure-vars"></a>
### 1.2. Configure Environment Variables

This is the most critical step. These variables are used in subsequent commands to target the correct resources.

```bash
export PROJECT_ID=<PROJECT_ID>
export REGION=<REGION_for_cloud_build>
export CLUSTER_REGION=<REGION_of_your_cluster>
export CLUSTER_NAME=<YOUR_GKE_CLUSTER_NAME>
export KUEUE_NAME=<YOUR_KUEUE_NAME>
export ARTIFACT_REGISTRY=<your-artifact-registry-repo-full-path>
export GCS_BUCKET=<your-gcs-bucket-for-logs>
export SGLANG_IMAGE=lmsysorg/sglang:dev
export SGLANG_VERSION=blackwell

# Set the project for gcloud commands
gcloud config set project $PROJECT_ID
```

Replace the following values:

| Variable              | Description                                                                                             | Example                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `PROJECT_ID` | Your Google Cloud Project ID. | `gcp-project-12345` |
| `REGION` | The GCP region to run the Cloud Build job. | `us-central1` |
| `CLUSTER_REGION` | The GCP region where your GKE cluster is located. | `us-central1` |
| `CLUSTER_NAME` | The name of your GKE cluster. | `a4x-gke-cluster` |
| `KUEUE_NAME` | The name of the Kueue local queue. The default queue created by the cluster toolkit is `a4`. Verify the name in your cluster. | `a4` |
| `ARTIFACT_REGISTRY` | Full path to your Artifact Registry repository. | `us-central1-docker.pkg.dev/gcp-project-12345/my-repo` |
| `GCS_BUCKET` | Name of your GCS bucket (do not include `gs://`). | `my-benchmark-logs-bucket` |
| `SGLANG_IMAGE` | The name for the Docker image to be built. | `lmsysorg/sglang:dev` |
| `SGLANG_VERSION` | The tag/version for the Docker image. | `blackwell` |


<a name="connect-cluster"></a>
### 1.3. Connect to your GKE Cluster

Fetch credentials for `kubectl` to communicate with your cluster.

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

<a name="get-hf-token"></a>
### 1.4. Get Hugging Face token

To access models through Hugging Face, you'll need a Hugging Face token.
  1.  Create a [Hugging Face account](https://huggingface.co/) if you don't have one.
  2.  For **gated models** ensure you have requested and been granted access on Hugging Face before proceeding.
  3.  Generate an Access Token: Go to **Your Profile > Settings > Access Tokens**.
  4.  Select **New Token**.
  5.  Specify a Name and a Role of at least `Read`.
  6.  Select **Generate a token**.
  7.  Copy the generated token to your clipboard. You'll use this later.


<a name="setup-hf-secret"></a>
### 1.5. Create Hugging Face Kubernetes Secret

Create a Kubernetes Secret with your Hugging Face token to enable the job to download model checkpoints from Hugging Face.

```bash
# Paste your Hugging Face token here
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

kubectl create secret generic hf-secret \
--from-literal=hf_api_token=${HF_TOKEN} \
--dry-run=client -o yaml | kubectl apply -f -
```

<a name="build-image"></a>
### 1.6. Build the SGLang Serving Image

This step uses Cloud Build to create a custom Docker image with SGLang and push it to your Artifact Registry repository.

> [!NOTE]
> This build process can take **up to 30 minutes** as it compiles and installs several dependencies.

```bash
cd $REPO_ROOT/src/docker/sglang
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_SGLANG_IMAGE=$SGLANG_IMAGE,_SGLANG_VERSION=$SGLANG_VERSION \
    --timeout "2h" \
    --machine-type=a4x-highgpu-4g \
    --disk-size=1000
```

Optionally, you can monitor the build progress by streaming its logs. Replace `<BUILD_ID>` with the ID from the previous command's output.

```bash
BUILD_ID=<BUILD_ID>
gcloud builds log $BUILD_ID --stream --region=$REGION
```

> [!WARNING]
> You may see `pip's dependency resolver` warnings in the build logs. These are generally safe to ignore as long as the Cloud Build job completes successfully.

**You have now completed the environment setup!** You are ready to deploy a model.

<a name="run-the-recipe"></a>

## Run Benchmarks

Running the benchmarks with different configurations.

```bash
#Benchmark with 1gpu 81 frames
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."     --save-output --num-gpus 1 --num-frames 81
```
```bash
#Benchmark with 4gpu 81 frames
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."     --save-output --num-gpus 4 --tp_size 4 --num-frames 93

```

```bash
#Download image from internet to benchmark I2V model
#Benchmark with 1gpu 81 frames
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path sample_image.jpg --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "A curious raccoon"     --save-output --num-gpus 1 --num-frames 81
```

```bash
#Benchmark with 4gpu 93 frames
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path sample_image.jpg --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "A curious raccoon"     --save-output --num-gpus 4 --tp-size 4 --num-frames 93

```
### 2. Delete the VM

This command will delete the GCE instance and all its disks.

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```
.

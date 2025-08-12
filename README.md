# 🖼️ Image Inpainting with SAM + Stable Diffusion
This repository provides an image inpainting pipeline combining Segment Anything Model (SAM) and Stable Diffusion. You can either fill a masked region with new content or replace an object based on a text prompt and point coordinates.

<img width="2588" height="1406" alt="image" src="https://github.com/user-attachments/assets/9a0f9d2a-bbae-4179-bdc1-af48fea1c7cc" />

## Segment Anything
**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

<img width="2412" height="514" alt="image" src="https://github.com/user-attachments/assets/70253c9a-bd62-4ae9-9c46-a0a04993f778" />

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<img width="1057" height="705" alt="image" src="https://github.com/user-attachments/assets/66d4dcc8-31dc-475c-a9bc-fa25fbba1891" />

## Stable Diffusion v2
Stable Diffusion v2 refers to a specific configuration of the model architecture that uses a downsampling-factor 8 autoencoder with an 865M UNet and OpenCLIP ViT-H/14 text encoder for the diffusion model. The SD 2-v model produces 768x768 px outputs.

Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) and 50 DDIM sampling steps show the relative improvements of the checkpoints:

<img width="936" height="576" alt="image" src="https://github.com/user-attachments/assets/bc89eb6a-2f2c-4478-bf8b-e0998ac31d84" />

### Text-to-Image
<img width="2560" height="512" alt="image" src="https://github.com/user-attachments/assets/599b91ba-e076-4d41-a047-113e13cc9dc0" />

Stable Diffusion 2 is a latent diffusion model conditioned on the penultimate text embeddings of a CLIP ViT-H/14 text encoder. 

## 🔧 Requirements
### 1. Create and Activate Conda Environment
```
conda create -n inpaint python=3.11 -y
conda activate inpaint
```
### 2. Install Dependencies
```
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install diffusers transformers accelerate scipy safetensors
```

### 📥 Download Pretrained Checkpoints
Download the SAM ViT-H checkpoint and place it inside the pretrained/ directory:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 🚀 Run the Inpainting Script
#### Fill Example:
```
python fill_anything.py \
    --input_img ./examples/fill-anything/sample1.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained/sam_vit_h_4b8939.pth
```
#### Replace Example:
```
python replace_anything.py \
    --input_img ./examples/replace-anything/dog.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "sit on the swing" \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained/sam_vit_h_4b8939.pth
```

# 🌐 Image Inpainting Web Application

A modern web interface for the Image Inpainting project using SAM (Segment Anything Model) + Stable Diffusion. This application provides an intuitive way to fill or replace parts of images using AI.

## 🎯 Features

- **Interactive Image Upload**: Drag & drop or click to upload images
- **Point-and-Click Selection**: Click on the image to select the area for inpainting
- **Two Inpainting Modes**:
  - **Fill**: Fill a selected area with new content based on text prompt
  - **Replace**: Replace an object with something else based on text prompt
- **Real-time Preview**: See your selected coordinates and mask
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, gradient-based design with smooth animations

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Segment Anything Model
pip install -e segment_anything
```

### 2. Download SAM Checkpoint

```bash
# Download the SAM ViT-H checkpoint (2.6GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Move it to the pretrained directory
mv sam_vit_h_4b8939.pth pretrained/
```

### 3. Start the Server

```bash
# Using the startup script (recommended)
python run_server.py

# Or directly with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Your Browser

Navigate to: `http://localhost:8000`

## 📱 How to Use

### Step 1: Upload an Image
- Click "Choose Image" or drag & drop an image file
- Supported formats: JPG, PNG, JPEG
- The image will be displayed on a canvas

### Step 2: Select a Point
- Click anywhere on the image to select the area you want to modify
- You'll see a red crosshair indicating your selection
- The coordinates will be displayed below the image

### Step 3: Enter a Text Prompt
- Describe what you want to fill or replace the selected area with
- Examples:
  - For Fill: "a beautiful garden", "blue sky with clouds"
  - For Replace: "a red car", "a person sitting"

### Step 4: Choose Operation
- **Fill**: Fills the selected area with new content
- **Replace**: Replaces the selected object with something new

### Step 5: View Results
- The processing will take 1-3 minutes depending on your hardware
- You'll see three images: Original, Mask, and Result
- Click "Process New Image" to start over

## ⚙️ Configuration Options

### Dilate Kernel Size
- Controls how much the mask is expanded around the selected point
- Range: 1-50 pixels
- Default: 15
- Higher values = larger affected area

### Advanced Settings
You can modify these in [`app.py`](app.py):

```python
# Model configuration
SAM_MODEL_TYPE = "vit_h"  # Options: vit_h, vit_l, vit_b
SAM_CHECKPOINT = "./pretrained/sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 🏗️ Architecture

### Backend (FastAPI)
- [`app.py`](app.py): Main FastAPI application
- **Endpoints**:
  - `GET /`: Serves the web interface
  - `POST /api/fill`: Fill operation
  - `POST /api/replace`: Replace operation
  - `GET /api/health`: Health check

### Frontend
- [`static/style.css`](static/style.css): Modern CSS with gradients and animations
- [`static/script.js`](static/script.js): Interactive JavaScript for image handling
- **Features**:
  - Canvas-based image display
  - Click coordinate detection
  - Drag & drop file upload
  - Responsive design
  - Loading animations

### Core Processing
- [`main_fill.py`](main_fill.py): Fill operation logic
- [`main_replace.py`](main_replace.py): Replace operation logic
- [`utils/`](utils/): Utility functions for image processing

## 🔧 API Reference

### Fill Endpoint
```http
POST /api/fill
Content-Type: multipart/form-data

Parameters:
- image: Image file
- point_x: X coordinate (float)
- point_y: Y coordinate (float)
- text_prompt: Description text (string)
- dilate_kernel_size: Mask dilation (int, default: 15)
```

### Replace Endpoint
```http
POST /api/replace
Content-Type: multipart/form-data

Parameters:
- image: Image file
- point_x: X coordinate (float)
- point_y: Y coordinate (float)
- text_prompt: Description text (string)
- dilate_kernel_size: Mask dilation (int, default: 15)
```

### Response Format
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "result": "data:image/png;base64,..."
}
```

## 🎨 Customization

### Styling
Modify [`static/style.css`](static/style.css) to change:
- Color schemes
- Layout
- Animations
- Responsive breakpoints

### Functionality
Modify [`static/script.js`](static/script.js) to add:
- Multiple point selection
- Batch processing
- Additional image filters
- Custom UI components

## 🚨 Troubleshooting

### Common Issues

**1. "SAM checkpoint not found"**
```bash
# Download the checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth pretrained/
```

**2. "CUDA out of memory"**
- The app automatically falls back to CPU if CUDA is unavailable
- For CPU-only usage, processing will be slower but still functional

**3. "Module not found" errors**
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -e segment_anything
```

**4. Slow processing**
- First run downloads Stable Diffusion models (~5GB)
- Subsequent runs are faster
- CPU processing takes 2-5 minutes per image
- GPU processing takes 30-60 seconds per image

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster processing
2. **Image Size**: Smaller images process faster
3. **Batch Processing**: Process multiple images in sequence
4. **Model Caching**: Models are cached after first use

## 📊 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 10GB free disk space
- CPU: Any modern processor

### Recommended Requirements
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 20GB free disk space
- CPU: Intel i7 or AMD Ryzen 7+

# 🐳 Docker & Kubernetes Deployment Guide

This guide provides comprehensive instructions for deploying the Image Inpainting application using Docker and Kubernetes.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Helm Deployment](#helm-deployment)
- [Configuration](#configuration)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)
- [Production Considerations](#production-considerations)

## 🔧 Prerequisites

### System Requirements
- **Docker**: Version 20.10+ with GPU support
- **Kubernetes**: Version 1.20+
- **kubectl**: Latest version
- **Helm**: Version 3.0+ (optional)
- **NVIDIA GPU**: With CUDA 11.8+ support
- **Storage**: 50GB+ available space

### GPU Support
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🐳 Docker Deployment

### 1. Build the Docker Image

```bash
# Build the image
docker build -t image-inpainting:latest .

# Build with specific tag
docker build -t image-inpainting:v1.0.0 .
```

### 2. Run with Docker Compose (Recommended)

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### 3. Manual Docker Run

```bash
# Create necessary directories
mkdir -p uploads results pretrained

# Download SAM checkpoint
wget -O pretrained/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Run the container
docker run -d \
  --name image-inpainting \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/pretrained:/app/pretrained \
  -e CUDA_VISIBLE_DEVICES=0 \
  image-inpainting:latest
```

### 4. Docker Commands

```bash
# View running containers
docker ps

# View logs
docker logs image-inpainting

# Execute commands in container
docker exec -it image-inpainting bash

# Stop container
docker stop image-inpainting

# Remove container
docker rm image-inpainting
```

## ☸️ Kubernetes Deployment

### 1. Prepare Your Cluster

```bash
# Verify cluster connection
kubectl cluster-info

# Create namespace (optional)
kubectl create namespace image-inpainting

# Set default namespace
kubectl config set-context --current --namespace=image-inpainting
```

### 2. Deploy with Raw Manifests

```bash
# Apply all manifests
kubectl apply -f k8s/

# Or apply individually
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Deploy with Kustomize

```bash
# Deploy using kustomize
kubectl apply -k k8s/

# View generated resources
kubectl kustomize k8s/
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -l app=image-inpainting

# Check services
kubectl get services

# Check persistent volumes
kubectl get pv,pvc

# View logs
kubectl logs -l app=image-inpainting -f
```

### 5. Access the Application

```bash
# Port forward for local access
kubectl port-forward service/image-inpainting-service 8000:8000

# Or use NodePort
kubectl get service image-inpainting-nodeport
# Access via http://NODE_IP:30800

# Or use LoadBalancer (cloud environments)
kubectl get service image-inpainting-loadbalancer
```

## ⛵ Helm Deployment

### 1. Install Helm Chart

```bash
# Install from local chart
helm install image-inpainting k8s/helm/image-inpainting/

# Install with custom values
helm install image-inpainting k8s/helm/image-inpainting/ \
  --values k8s/helm/image-inpainting/values.yaml

# Install with overrides
helm install image-inpainting k8s/helm/image-inpainting/ \
  --set image.tag=v1.0.0 \
  --set replicaCount=2
```

### 2. Helm Management

```bash
# List releases
helm list

# Upgrade release
helm upgrade image-inpainting k8s/helm/image-inpainting/

# Rollback release
helm rollback image-inpainting 1

# Uninstall release
helm uninstall image-inpainting
```

### 3. Custom Values

Create a custom `values.yaml`:

```yaml
# custom-values.yaml
replicaCount: 2

image:
  tag: "v1.0.0"

resources:
  limits:
    memory: "16Gi"
    cpu: "4000m"

ingress:
  enabled: true
  hosts:
    - host: inpainting.yourdomain.com
      paths:
        - path: /
          pathType: Prefix

persistence:
  results:
    size: 100Gi
```

```bash
helm install image-inpainting k8s/helm/image-inpainting/ -f custom-values.yaml
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SAM_MODEL_TYPE` | SAM model type | `vit_h` |
| `SAM_CHECKPOINT` | Path to SAM checkpoint | `/app/pretrained/sam_vit_h_4b8939.pth` |
| `DEVICE` | Compute device | `cuda` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MAX_WORKERS` | Worker processes | `4` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Storage Configuration

#### Persistent Volumes
- **uploads**: 10GB for uploaded images
- **results**: 50GB for processed results
- **pretrained**: 20GB for model checkpoints

#### Storage Classes
```yaml
# Example storage class for fast SSD
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
```

### GPU Configuration

#### Node Labels
```bash
# Label GPU nodes
kubectl label nodes <node-name> accelerator=nvidia-tesla-gpu
```

#### GPU Operator (NVIDIA)
```bash
# Install GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator-resources \
  --create-namespace
```

## 📊 Monitoring & Troubleshooting

### Health Checks

```bash
# Check application health
curl http://localhost:8000/api/health

# Kubernetes health check
kubectl get pods -l app=image-inpainting
kubectl describe pod <pod-name>
```

### Common Issues

#### 1. GPU Not Available
```bash
# Check GPU availability
kubectl describe node <gpu-node>
kubectl get nodes -l accelerator=nvidia-tesla-gpu

# Verify GPU operator
kubectl get pods -n gpu-operator-resources
```

#### 2. Out of Memory
```bash
# Check resource usage
kubectl top pods
kubectl describe pod <pod-name>

# Increase memory limits
kubectl patch deployment image-inpainting-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"image-inpainting","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
```

#### 3. Storage Issues
```bash
# Check PVC status
kubectl get pvc
kubectl describe pvc <pvc-name>

# Check available storage
kubectl exec -it <pod-name> -- df -h
```

### Logging

```bash
# View application logs
kubectl logs -l app=image-inpainting -f

# View logs from specific container
kubectl logs <pod-name> -c image-inpainting

# Export logs
kubectl logs <pod-name> > app.log
```

### Debugging

```bash
# Access container shell
kubectl exec -it <pod-name> -- bash

# Run debug commands
kubectl exec -it <pod-name> -- python -c "import torch; print(torch.cuda.is_available())"

# Port forward for debugging
kubectl port-forward <pod-name> 8000:8000
```

## 🚀 Production Considerations

### Security

#### 1. Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: image-inpainting-netpol
spec:
  podSelector:
    matchLabels:
      app: image-inpainting
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

#### 2. Pod Security Standards
```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: image-inpainting
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### High Availability

#### 1. Multiple Replicas
```yaml
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
```

#### 2. Pod Disruption Budget
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: image-inpainting-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: image-inpainting
```

### Performance Optimization

#### 1. Resource Requests/Limits
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

#### 2. Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: image-inpainting-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: image-inpainting-app
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Backup & Recovery

#### 1. Persistent Volume Snapshots
```bash
# Create volume snapshot
kubectl create -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: results-snapshot
spec:
  source:
    persistentVolumeClaimName: results-pvc
EOF
```

#### 2. Application Backup
```bash
# Backup configuration
kubectl get configmap image-inpainting-config -o yaml > config-backup.yaml
kubectl get secret image-inpainting-secrets -o yaml > secrets-backup.yaml
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t image-inpainting:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        docker tag image-inpainting:${{ github.sha }} your-registry/image-inpainting:${{ github.sha }}
        docker push your-registry/image-inpainting:${{ github.sha }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/image-inpainting-app \
          image-inpainting=your-registry/image-inpainting:${{ github.sha }}
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

## 🆘 Support

For issues and questions:
1. Check the [troubleshooting section](#monitoring--troubleshooting)
2. Review application logs
3. Check Kubernetes events: `kubectl get events`
4. Open an issue in the project repository
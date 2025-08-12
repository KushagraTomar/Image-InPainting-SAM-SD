# 🖼️ Image Inpainting with SAM + Stable Diffusion
This repository provides an image inpainting pipeline combining Segment Anything Model (SAM) and Stable Diffusion. You can either fill a masked region with new content or replace an object based on a text prompt and point coordinates.

<img width="2588" height="1406" alt="image" src="https://github.com/user-attachments/assets/9a0f9d2a-bbae-4179-bdc1-af48fea1c7cc" />

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

## 🌐 Web Application

For a user-friendly web interface, see the [Web Application Guide](WEB_README.md).

### Quick Start (Web Interface)
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e segment_anything

# Download SAM checkpoint
wget -O pretrained/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Start the web server
python run_server.py
# Open http://localhost:8000 in your browser
```

## 🐳 Docker & Kubernetes Deployment

This project supports containerized deployment with Docker and Kubernetes for production environments.

### Quick Docker Start
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the application at http://localhost:8000
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/

# Or use Helm
helm install image-inpainting k8s/helm/image-inpainting/
```

📖 **For detailed Docker and Kubernetes deployment instructions, see [DOCKER_KUBERNETES_README.md](DOCKER_KUBERNETES_README.md)**

## 📁 Project Structure

```
├── app.py                          # FastAPI web application
├── run_server.py                   # Server startup script
├── main_fill.py                    # Fill operation logic
├── main_replace.py                 # Replace operation logic
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container definition
├── docker-compose.yml              # Docker Compose configuration
├── .dockerignore                   # Docker ignore file
├── k8s/                           # Kubernetes manifests
│   ├── deployment.yaml            # Kubernetes deployment
│   ├── service.yaml               # Kubernetes services
│   ├── configmap.yaml             # Configuration and secrets
│   ├── kustomization.yaml         # Kustomize configuration
│   └── helm/                      # Helm chart
│       └── image-inpainting/      # Helm chart files
├── static/                        # Web UI assets
│   ├── style.css                  # Web interface styles
│   └── script.js                  # Web interface logic
├── utils/                         # Utility functions
├── segment_anything/              # SAM model implementation
├── examples/                      # Example images
├── pretrained/                    # Model checkpoints
├── uploads/                       # Uploaded images
└── results/                       # Processing results
```

## 🚀 Deployment Options

### 1. Local Development
- **Command Line**: Use `python main_fill.py` or `python main_replace.py`
- **Web Interface**: Use `python run_server.py` for local web UI

### 2. Docker (Recommended for Production)
- **Single Container**: `docker run` with GPU support
- **Docker Compose**: Multi-service setup with volumes and networking
- **Supports**: GPU acceleration, persistent storage, health checks

### 3. Kubernetes (Enterprise/Scale)
- **Raw Manifests**: Direct kubectl deployment
- **Kustomize**: Configuration management
- **Helm Charts**: Package management and templating
- **Features**: Auto-scaling, load balancing, persistent volumes, monitoring

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `SAM_MODEL_TYPE` | SAM model variant | `vit_h` |
| `SAM_CHECKPOINT` | Path to SAM weights | `./pretrained/sam_vit_h_4b8939.pth` |
| `DEVICE` | Compute device | `cuda` if available |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slow)
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage**: 20GB+ for models and processing

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
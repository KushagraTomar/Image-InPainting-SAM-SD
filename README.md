# 🖼️ Image Inpainting with SAM + Stable Diffusion
This repository provides an image inpainting pipeline combining Segment Anything Model (SAM) and Stable Diffusion. You can either fill a masked region with new content or replace an object based on a text prompt and point coordinates.
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

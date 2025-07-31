# 🖼️ Image Inpainting with SAM + Stable Diffusion
This repository provides an image inpainting pipeline combining Segment Anything Model (SAM) and Stable Diffusion. You can either fill a masked region with new content or replace an object based on a text prompt and point coordinates.

<img width="2588" height="1406" alt="image" src="https://github.com/user-attachments/assets/9a0f9d2a-bbae-4179-bdc1-af48fea1c7cc" />

#### Segment Anything
**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

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

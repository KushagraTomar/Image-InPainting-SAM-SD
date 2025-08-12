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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project extends the original Image Inpainting project. Please refer to the main [`LICENSE`](LICENSE) file for details.

## 🙏 Acknowledgments

- **Segment Anything Model (SAM)** by Meta AI
- **Stable Diffusion** by Stability AI
- **FastAPI** for the web framework
- Original command-line implementation authors
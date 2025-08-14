import os
import io
import uuid
import base64
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import existing functionality
from main_fill import (
    load_img_to_array, save_array_to_img, dilate_mask, 
    fill_img_with_sd, predict_masks_with_sam
)
from main_replace import replace_img_with_sd

app = FastAPI(title="Image Inpainting API", description="SAM + Stable Diffusion Image Inpainting")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "./pretrained/sam_vit_b_01ec64.pth"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

class ImageInpaintingService:
    def __init__(self):
        self.sam_model_type = SAM_MODEL_TYPE
        self.sam_checkpoint = SAM_CHECKPOINT
        self.device = DEVICE
    
    def process_fill(
        self, 
        image: np.ndarray, 
        point_coords: List[float], 
        point_labels: List[int],
        text_prompt: str,
        dilate_kernel_size: Optional[int] = 15
    ):
        """Process fill operation"""
        try:
            # Predict masks with SAM
            masks, scores, logits = predict_masks_with_sam(
                image,
                [point_coords],
                point_labels,
                model_type=self.sam_model_type,
                ckpt_p=self.sam_checkpoint,
                device=self.device,
            )
            
            # Convert masks to binary
            masks = masks.astype(np.uint8) * 255
            
            # Dilate mask if specified
            if dilate_kernel_size:
                masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
            
            # Use the best mask (first one, highest score)
            best_mask = masks[0]
            
            # Fill the image
            filled_image = fill_img_with_sd(
                image, best_mask, text_prompt, device=self.device
            )
            
            return filled_image, best_mask
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Fill processing failed: {str(e)}")
    
    def process_replace(
        self, 
        image: np.ndarray, 
        point_coords: List[float], 
        point_labels: List[int],
        text_prompt: str,
        dilate_kernel_size: Optional[int] = 15
    ):
        """Process replace operation"""
        try:
            # Predict masks with SAM
            masks, scores, logits = predict_masks_with_sam(
                image,
                [point_coords],
                point_labels,
                model_type=self.sam_model_type,
                ckpt_p=self.sam_checkpoint,
                device=self.device,
            )
            
            # Convert masks to binary
            masks = masks.astype(np.uint8) * 255
            
            # Dilate mask if specified
            if dilate_kernel_size:
                masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
            
            # Use the best mask (first one, highest score)
            best_mask = masks[0]
            
            # Replace the image
            replaced_image = replace_img_with_sd(
                image, best_mask, text_prompt, device=self.device
            )
            
            return replaced_image, best_mask
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Replace processing failed: {str(e)}")

# Initialize service
inpainting_service = ImageInpaintingService()

def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Inpainting</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="container">
            <h1>🖼️ Image Inpainting with SAM + Stable Diffusion</h1>
            <div class="upload-section">
                <input type="file" id="imageInput" accept="image/*" />
                <label for="imageInput" class="upload-btn">Choose Image</label>
            </div>
            
            <div id="imageContainer" class="image-container" style="display: none;">
                <canvas id="imageCanvas"></canvas>
                <div class="coordinates">
                    <span>Click coordinates: </span>
                    <span id="coordinates">None</span>
                </div>
            </div>
            
            <div id="controlsSection" class="controls-section" style="display: none;">
                <div class="form-group">
                    <label for="textPrompt">Text Prompt:</label>
                    <input type="text" id="textPrompt" placeholder="Enter description for inpainting..." />
                </div>
                
                <div class="form-group">
                    <label for="dilateSize">Dilate Kernel Size:</label>
                    <input type="number" id="dilateSize" value="15" min="1" max="50" />
                </div>
                
                <div class="button-group">
                    <button id="fillBtn" class="action-btn fill-btn">Fill</button>
                    <button id="replaceBtn" class="action-btn replace-btn">Replace</button>
                </div>
            </div>
            
            <div id="loadingSection" class="loading-section" style="display: none;">
                <div class="spinner"></div>
                <p>Processing image... This may take a few minutes.</p>
            </div>
            
            <div id="resultsSection" class="results-section" style="display: none;">
                <h3>Results</h3>
                <div class="result-images">
                    <div class="result-item">
                        <h4>Original</h4>
                        <img id="originalResult" />
                    </div>
                    <div class="result-item">
                        <h4>Mask</h4>
                        <img id="maskResult" />
                    </div>
                    <div class="result-item">
                        <h4>Result</h4>
                        <img id="finalResult" />
                    </div>
                </div>
                <button id="newImageBtn" class="action-btn">Process New Image</button>
            </div>
        </div>
        
        <script src="/static/script.js"></script>
    </body>
    </html>
    """

@app.post("/api/fill")
async def fill_image(
    image: UploadFile = File(...),
    point_x: float = Form(...),
    point_y: float = Form(...),
    text_prompt: str = Form(...),
    dilate_kernel_size: int = Form(15)
):
    """Fill operation endpoint"""
    try:
        # Read and process image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        if image_pil.mode == "RGBA":
            image_pil = image_pil.convert("RGB")
        image_array = np.array(image_pil)
        
        # Process fill
        result_image, mask = inpainting_service.process_fill(
            image_array,
            [point_x, point_y],
            [1],  # positive label
            text_prompt,
            dilate_kernel_size
        )
        
        # Convert results to base64
        original_b64 = image_to_base64(image_array)
        mask_b64 = image_to_base64(mask)
        result_b64 = image_to_base64(result_image)
        
        return JSONResponse({
            "success": True,
            "original": original_b64,
            "mask": mask_b64,
            "result": result_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/replace")
async def replace_image(
    image: UploadFile = File(...),
    point_x: float = Form(...),
    point_y: float = Form(...),
    text_prompt: str = Form(...),
    dilate_kernel_size: int = Form(15)
):
    """Replace operation endpoint"""
    try:
        # Read and process image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        if image_pil.mode == "RGBA":
            image_pil = image_pil.convert("RGB")
        image_array = np.array(image_pil)
        
        # Process replace
        result_image, mask = inpainting_service.process_replace(
            image_array,
            [point_x, point_y],
            [1],  # positive label
            text_prompt,
            dilate_kernel_size
        )
        
        # Convert results to base64
        original_b64 = image_to_base64(image_array)
        mask_b64 = image_to_base64(mask)
        result_b64 = image_to_base64(result_image)
        
        return JSONResponse({
            "success": True,
            "original": original_b64,
            "mask": mask_b64,
            "result": result_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "sam_model": SAM_MODEL_TYPE,
        "sam_checkpoint_exists": os.path.exists(SAM_CHECKPOINT)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
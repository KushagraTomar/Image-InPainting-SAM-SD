#!/usr/bin/env python3
"""
Startup script for the Image Inpainting Web Application
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if SAM checkpoint exists
    sam_checkpoint = Path("pretrained/sam_vit_b_01ec64.pth")
    if not sam_checkpoint.exists():
        print("❌ SAM checkpoint not found!")
        print("Please download the SAM checkpoint:")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print("And place it in the pretrained/ directory")
        return False
    
    # Check if segment_anything is installed
    try:
        import segment_anything
        print("✅ Segment Anything Model found")
    except ImportError:
        print("❌ Segment Anything Model not installed!")
        print("Please install it with: pip install -e segment_anything")
        return False
    
    # Check if required packages are installed
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'diffusers', 
        'transformers', 'PIL', 'numpy', 'cv2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package} found")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not found")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results', 'static']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def main():
    print("🚀 Starting Image Inpainting Web Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n✅ All checks passed!")
    print("🌐 Starting web server...")
    print("Open your browser and go to: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
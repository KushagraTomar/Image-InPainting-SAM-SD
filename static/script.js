class ImageInpaintingApp {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.currentImage = null;
        this.clickCoordinates = null;
        this.isProcessing = false;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        this.imageInput = document.getElementById('imageInput');
        this.imageContainer = document.getElementById('imageContainer');
        this.imageCanvas = document.getElementById('imageCanvas');
        this.coordinatesDisplay = document.getElementById('coordinates');
        this.controlsSection = document.getElementById('controlsSection');
        this.textPrompt = document.getElementById('textPrompt');
        this.dilateSize = document.getElementById('dilateSize');
        this.fillBtn = document.getElementById('fillBtn');
        this.replaceBtn = document.getElementById('replaceBtn');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.newImageBtn = document.getElementById('newImageBtn');
        this.originalResult = document.getElementById('originalResult');
        this.maskResult = document.getElementById('maskResult');
        this.finalResult = document.getElementById('finalResult');
        
        this.canvas = this.imageCanvas;
        this.ctx = this.canvas.getContext('2d');
    }
    
    bindEvents() {
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.fillBtn.addEventListener('click', () => this.processImage('fill'));
        this.replaceBtn.addEventListener('click', () => this.processImage('replace'));
        this.newImageBtn.addEventListener('click', () => this.resetApp());
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.displayImage(img);
                this.showSection(this.imageContainer);
                this.showSection(this.controlsSection);
                this.resetCoordinates();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    displayImage(img) {
        // Calculate canvas size to fit image while maintaining aspect ratio
        const maxWidth = 800;
        const maxHeight = 500;
        
        let { width, height } = img;
        
        if (width > maxWidth) {
            height = (height * maxWidth) / width;
            width = maxWidth;
        }
        
        if (height > maxHeight) {
            width = (width * maxHeight) / height;
            height = maxHeight;
        }
        
        this.canvas.width = width;
        this.canvas.height = height;
        
        // Store scale factors for coordinate conversion
        this.scaleX = img.width / width;
        this.scaleY = img.height / height;
        
        this.ctx.drawImage(img, 0, 0, width, height);
    }
    
    handleCanvasClick(event) {
        if (this.isProcessing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Convert to original image coordinates
        const originalX = Math.round(x * this.scaleX);
        const originalY = Math.round(y * this.scaleY);
        
        this.clickCoordinates = { x: originalX, y: originalY };
        this.coordinatesDisplay.textContent = `(${originalX}, ${originalY})`;
        
        // Redraw image and add click indicator
        this.ctx.drawImage(this.currentImage, 0, 0, this.canvas.width, this.canvas.height);
        this.drawClickIndicator(x, y);
        
        // Enable action buttons
        this.fillBtn.disabled = false;
        this.replaceBtn.disabled = false;
    }
    
    drawClickIndicator(x, y) {
        this.ctx.save();
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.fillStyle = '#ff0000';
        this.ctx.lineWidth = 3;
        
        // Draw crosshair
        this.ctx.beginPath();
        this.ctx.moveTo(x - 10, y);
        this.ctx.lineTo(x + 10, y);
        this.ctx.moveTo(x, y - 10);
        this.ctx.lineTo(x, y + 10);
        this.ctx.stroke();
        
        // Draw circle
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.fill();
        
        this.ctx.restore();
    }
    
    async processImage(operation) {
        if (!this.clickCoordinates || !this.textPrompt.value.trim()) {
            alert('Please select a point on the image and enter a text prompt.');
            return;
        }
        
        this.isProcessing = true;
        this.showSection(this.loadingSection);
        this.hideSection(this.controlsSection);
        this.fillBtn.disabled = true;
        this.replaceBtn.disabled = true;
        
        try {
            const formData = new FormData();
            
            // Convert canvas to blob
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = this.currentImage.width;
            canvas.height = this.currentImage.height;
            ctx.drawImage(this.currentImage, 0, 0);
            
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
            
            formData.append('image', blob, 'image.png');
            formData.append('point_x', this.clickCoordinates.x);
            formData.append('point_y', this.clickCoordinates.y);
            formData.append('text_prompt', this.textPrompt.value.trim());
            formData.append('dilate_kernel_size', this.dilateSize.value);
            
            const response = await fetch(`/api/${operation}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Processing failed');
            }
            
            const result = await response.json();
            this.displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
            this.hideSection(this.loadingSection);
            this.showSection(this.controlsSection);
        } finally {
            this.isProcessing = false;
        }
    }
    
    displayResults(result) {
        this.originalResult.src = result.original;
        this.maskResult.src = result.mask;
        this.finalResult.src = result.result;
        
        this.hideSection(this.loadingSection);
        this.showSection(this.resultsSection);
        
        // Add fade-in animation
        this.resultsSection.classList.add('fade-in');
    }
    
    resetApp() {
        // Reset all state
        this.currentImage = null;
        this.clickCoordinates = null;
        this.isProcessing = false;
        
        // Clear form
        this.imageInput.value = '';
        this.textPrompt.value = '';
        this.dilateSize.value = '15';
        
        // Reset UI
        this.resetCoordinates();
        this.hideSection(this.imageContainer);
        this.hideSection(this.controlsSection);
        this.hideSection(this.loadingSection);
        this.hideSection(this.resultsSection);
        
        // Clear canvas
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        
        // Reset buttons
        this.fillBtn.disabled = true;
        this.replaceBtn.disabled = true;
    }
    
    resetCoordinates() {
        this.clickCoordinates = null;
        this.coordinatesDisplay.textContent = 'None';
        this.fillBtn.disabled = true;
        this.replaceBtn.disabled = true;
    }
    
    showSection(element) {
        element.style.display = 'block';
    }
    
    hideSection(element) {
        element.style.display = 'none';
        element.classList.remove('fade-in');
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageInpaintingApp();
});

// Add some utility functions for better UX
document.addEventListener('DOMContentLoaded', () => {
    // Add drag and drop functionality
    const container = document.querySelector('.container');
    const imageInput = document.getElementById('imageInput');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        container.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        container.style.background = 'rgba(102, 126, 234, 0.1)';
    }
    
    function unhighlight(e) {
        container.style.background = 'white';
    }
    
    container.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            imageInput.files = files;
            const event = new Event('change', { bubbles: true });
            imageInput.dispatchEvent(event);
        }
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'o':
                    e.preventDefault();
                    imageInput.click();
                    break;
                case 'Enter':
                    e.preventDefault();
                    const fillBtn = document.getElementById('fillBtn');
                    if (!fillBtn.disabled) {
                        fillBtn.click();
                    }
                    break;
            }
        }
    });
});
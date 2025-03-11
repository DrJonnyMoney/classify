import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory

# Create Flask app
app = Flask(__name__)

# Print startup message
print("=" * 50)
print("Starting ResNet18 Inference Server")
print("=" * 50)

# Device detection
print("Detecting device...")
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

# Load pre-trained ResNet18 model
print("Loading ResNet18 model (this may take a while)...")
model_load_start = time.time()
model = models.resnet18(pretrained=True)
print(f"Model architecture loaded in {time.time() - model_load_start:.2f} seconds")

print("Setting model to evaluation mode...")
model.eval()

print("Moving model to device...")
device_transfer_start = time.time()
model = model.to(device)
print(f"Model transferred to {device} in {time.time() - device_transfer_start:.2f} seconds")
print(f"Total initialization time: {time.time() - start_time:.2f} seconds")
print("=" * 50)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet classes
def load_imagenet_classes():
    """Load ImageNet class labels"""
    cache_path = os.path.expanduser('~/.cache/imagenet_classes.txt')
    
    # Check if cached file exists
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return [line.strip() for line in f]
    
    # If no cache, use a minimal fallback list
    return [f"Class_{i}" for i in range(1000)]

classes = load_imagenet_classes()

# Define the predict route first so it takes precedence over the catch-all
@app.route('/predict', methods=['POST'])
@app.route('/notebook/<username>/<notebook_name>/predict', methods=['POST'])
def predict(username=None, notebook_name=None):
    """Process the uploaded image and return predictions"""
    print(f"Received prediction request at {time.strftime('%H:%M:%S')}")
    
    if 'file' not in request.files:
        print("Error: No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({'error': 'No selected file'})
    
    try:
        print(f"Processing file: {file.filename}")
        
        # Read the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(f"Image opened: {img.format}, size: {img.size}, mode: {img.mode}")
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            print("Converting RGBA to RGB")
            img = img.convert('RGB')
        
        # Preprocess the image
        print("Preprocessing image...")
        preprocess_start = time.time()
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        input_batch = input_batch.to(device)
        print(f"Preprocessing completed in {time.time() - preprocess_start:.2f} seconds")
        
        # Perform inference
        print("Running inference...")
        inference_start = time.time()
        with torch.no_grad():
            output = model(input_batch)
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        # Get top 5 predictions
        print("Processing results...")
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # Format results
        results = []
        for i in range(top5_prob.size(0)):
            class_index = top5_catid[i].item()
            if class_index < len(classes):
                class_name = classes[class_index]
            else:
                class_name = f"Unknown class (index {class_index})"
            
            results.append({
                'class': class_name,
                'probability': float(top5_prob[i].item()) * 100
            })
        
        print(f"Top prediction: {results[0]['class']} ({results[0]['probability']:.2f}%)")
        print(f"Total processing time: {time.time() - preprocess_start:.2f} seconds")
        
        return jsonify({
            'predictions': results,
            'processing_time': inference_time
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

# Catch-all route for Kubeflow URL patterns - positioned AFTER specific routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Handle routes that aren't covered by specific endpoints and render the main page"""
    # Exclude API endpoints from catch-all
    if path == 'predict':
        return jsonify({'error': 'Method not allowed'}), 405
        
    print(f"Caught path: {path}")
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    print("=" * 50)
    # Use port 8888 for Kubeflow compatibility
    app.run(host='0.0.0.0', port=8888, debug=False)

import torch
import numpy as np
from PIL import Image
import cv2

def load_model(model_path, device='cpu'):
    """
    Load the trained generator model
    """
    from model import GeneratorUNet
    
    model = GeneratorUNet(in_channels=3, out_channels=3, features=64)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'gen_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['gen_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess input image for the model
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    
    # Convert to tensor (C, H, W)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image

def postprocess_image(tensor):
    """
    Convert model output tensor to displayable image
    """
    # Denormalize from [-1, 1] to [0, 1]
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    image = (image + 1) / 2.0
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    return image

def create_sketch_from_image(image):
    """
    Convert an image to sketch-like appearance
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Invert and apply Gaussian blur
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert back and create sketch
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    
    # Convert to RGB for display
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    return sketch_rgb

def generate_image(model, input_image, device='cpu'):
    """
    Generate colored image from input sketch
    """
    with torch.no_grad():
        input_tensor = preprocess_image(input_image)
        input_tensor = input_tensor.to(device)
        
        output_tensor = model(input_tensor)
        output_image = postprocess_image(output_tensor)
        
    return output_image
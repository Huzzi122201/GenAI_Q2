import torch
import numpy as np
from PIL import Image
import cv2
import re


def _extract_state_dict(checkpoint):
    """Return the actual state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in (
            'gen_state_dict',
            'model_state_dict',
            'state_dict',
            'generator_state_dict',
            'generator',
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _remap_generator_keys(state_dict):
    """Map older training key names to the inference GeneratorUNet key names."""
    remapped = {}

    for key, value in state_dict.items():
        new_key = key

        # Remove common wrappers from DataParallel or nested modules.
        for prefix in ('module.', 'generator.', 'netG.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        # Map encoder/decoder naming to current model naming.
        new_key = re.sub(r'^enc([1-7])\.block\.', r'down\1.model.', new_key)
        new_key = re.sub(r'^dec([1-7])\.block\.', r'up\1.model.', new_key)

        # Keep bottleneck/final names aligned in case wrapped prefixes were removed.
        new_key = re.sub(r'^bottleneck\.', 'bottleneck.', new_key)
        new_key = re.sub(r'^final\.', 'final.', new_key)

        remapped[new_key] = value

    return remapped

def load_model(model_path, device='cpu'):
    """
    Load the trained generator model
    """
    from model import GeneratorUNet
    
    model = GeneratorUNet(in_channels=3, out_channels=3, features=64)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    # First try direct load for native checkpoints.
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Fallback for checkpoints trained with older naming conventions.
        remapped_state_dict = _remap_generator_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

        # Some training configs use bias=False for conv/transposed-conv layers,
        # so checkpoints may legitimately miss only bias tensors.
        only_bias_missing = all(k.endswith('.bias') for k in missing_keys)
        if only_bias_missing and missing_keys:
            named_params = dict(model.named_parameters())
            for key in missing_keys:
                if key in named_params:
                    with torch.no_grad():
                        named_params[key].zero_()

        # If keys other than biases are missing, surface a clear incompatibility error.
        if unexpected_keys or (missing_keys and not only_bias_missing):
            raise RuntimeError(
                "Checkpoint is incompatible with current GeneratorUNet architecture. "
                f"Missing keys: {missing_keys[:10]}... "
                f"Unexpected keys: {unexpected_keys[:10]}..."
            )
    
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
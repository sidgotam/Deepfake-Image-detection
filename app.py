# app.py
import os
import time
import uuid
from io import BytesIO
from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import SimpleCNN, ImprovedDeepfakeDetector
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "cnn_model.pth"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
CAM_OUTPUT_BASENAME = "cam_output"
CAM_OUTPUT_EXT = ".jpg"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load only trained checkpoint (no random/untrained fallback for prediction).
use_improved_model = True
model = None
model_loaded = False
model_info = {}

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model_type = checkpoint.get("model_type", "improved") if isinstance(checkpoint, dict) else "improved"

    load_errors = []
    if model_type == "improved":
        try:
            model = ImprovedDeepfakeDetector(num_classes=2, pretrained=False).to(device)
            model.load_state_dict(state_dict)
            use_improved_model = True
            model_loaded = True
        except Exception as e:
            load_errors.append(f"improved: {e}")
    elif model_type == "simple":
        try:
            model = SimpleCNN().to(device)
            model.load_state_dict(state_dict)
            use_improved_model = False
            model_loaded = True
        except Exception as e:
            load_errors.append(f"simple: {e}")

    # Backward compatibility with older checkpoints that may not have model_type.
    if not model_loaded:
        try:
            model = ImprovedDeepfakeDetector(num_classes=2, pretrained=False).to(device)
            model.load_state_dict(state_dict)
            use_improved_model = True
            model_loaded = True
        except Exception as e:
            load_errors.append(f"fallback improved: {e}")

    if not model_loaded:
        try:
            model = SimpleCNN().to(device)
            model.load_state_dict(state_dict)
            use_improved_model = False
            model_loaded = True
        except Exception as e:
            load_errors.append(f"fallback simple: {e}")

    if model_loaded:
        model.eval()
        model_info = checkpoint if isinstance(checkpoint, dict) else {}
        print(f"Loaded trained checkpoint from {MODEL_PATH}")
    else:
        print(f"Could not load trained checkpoint from {MODEL_PATH}. Errors: {load_errors}")
else:
    print(f"No checkpoint found at {MODEL_PATH}. Run `python train_model.py` first.")

# Transforms - use ImageNet normalization for ResNet, or custom for SimpleCNN
if use_improved_model:
    # ImageNet normalization for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_size = 224
else:
    # Custom normalization for SimpleCNN
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_size = 128

LABELS = {0: "Fake", 1: "Real"}

def preprocess_pil(img_pil):
    """Return a tensor ready for model and an RGB numpy image scaled 0..1 (for CAM)."""
    img = img_pil.convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # 1 x C x H x W
    # For cam overlay we need a float RGB numpy in [0,1]
    rgb_np = np.array(img.resize((img_size, img_size))).astype(np.float32) / 255.0
    return img_tensor.to(device), rgb_np
@app.route("/")
def home():
    """Serve the frontend page."""
    return send_from_directory("frontend", "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    """Serve other frontend assets (js, css, images)."""
    frontend_path = os.path.join("frontend", filename)
    if os.path.isfile(frontend_path):
        return send_from_directory("frontend", filename)
    return jsonify({"error": "Not found"}), 404
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded or model is None:
        return jsonify({
            "error": "Model is not trained/loaded. Train first with `python train_model.py`."
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (form field name must be 'image')"}), 400
    file = request.files["image"]
    # filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    # file.save(filepath)
    try:
        img_pil = Image.open(BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": "Cannot read image: " + str(e)}), 400

    try:
        input_tensor, rgb_np = preprocess_pil(img_pil)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)  # raw logits
            probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze()  # e.g. [p_fake, p_real]

        # Get predicted class with confidence threshold
        pred_idx = int(np.argmax(probs))
        max_prob = float(probs[pred_idx])
        
        # Confidence threshold - if below 60%, mark as uncertain
        confidence_threshold = 0.60
        if max_prob < confidence_threshold:
            pred_label = "Uncertain"
        else:
            pred_label = LABELS.get(pred_idx, str(pred_idx))

        # Percentages
        percent_fake = float(probs[0] * 100.0)
        percent_real = float(probs[1] * 100.0)

        # Grad-CAM: choose a safe target conv layer.
        import torch.nn as nn

        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("No Conv2d layers found for Grad-CAM.")

        # Prefer an architecture-specific layer when possible.
        target_layer = None
        if use_improved_model:
            # For ResNet18, use the last block conv2.
            try:
                target_layer = model.backbone.layer4[-1].conv2
            except Exception:
                target_layer = None

        # Fallback to the last conv layer.
        if target_layer is None:
            target_layer = conv_layers[-1]

        # pytorch-grad-cam expects gradients; ensure they are enabled.
        input_tensor = input_tensor.requires_grad_(True)
        with torch.enable_grad():
            # Clear any stale grads between requests.
            model.zero_grad(set_to_none=True)
            cam = GradCAM(model=model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # H x W, values 0..1
            # Explicitly release hooks to prevent occasional hook accumulation.
            if hasattr(cam, "activations_and_grads"):
                cam.activations_and_grads.release()

        if device.type == "cuda":
            # Ensure CUDA work is complete before saving the output.
            torch.cuda.synchronize()

        grayscale_cam = np.nan_to_num(grayscale_cam)
        grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

        # Overlay CAM on image.
        cam_image = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
        if cam_image is None:
            raise RuntimeError("Grad-CAM returned empty visualization.")

        # Save to a unique filename to avoid race conditions / partial writes.
        cam_filename = f"{CAM_OUTPUT_BASENAME}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{CAM_OUTPUT_EXT}"
        cam_path = os.path.join(STATIC_DIR, cam_filename)

        # Convert RGB->BGR for cv2, handling both float [0..1] and uint8.
        if cam_image.dtype != np.uint8:
            if cam_image.max() <= 1.0:
                cam_image = (cam_image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                cam_image = cam_image.clip(0, 255).astype(np.uint8)
        cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(cam_path, cam_bgr)
        if not ok:
            raise RuntimeError("Failed to write Grad-CAM image to disk.")

        heatmap_url = url_for("static", filename=cam_filename, _external=True)
        
        response = {
            "prediction": pred_label,
            "probabilities": {
                "Fake": round(percent_fake, 2),
                "Real": round(percent_real, 2)
            },
            "model": {
                "type": "ImprovedDeepfakeDetector" if use_improved_model else "SimpleCNN",
                "trained_epoch": model_info.get("epoch", None),
                "best_val_acc": model_info.get("val_acc", None)
            },
            # heatmap link served from static folder with cache-busting
            "heatmap": heatmap_url
        }

        return jsonify(response)
    except Exception as e:
        # Return error and print server side for debugging
        app.logger.exception("Prediction error")
        return jsonify({"error": "Prediction failed: " + str(e)}), 500

if __name__ == "__main__":
    # For local testing: run with `python app.py` then POST to /predict
    app.run(host="0.0.0.0", port=5000, debug=True)

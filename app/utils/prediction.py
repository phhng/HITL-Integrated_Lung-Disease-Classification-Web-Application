import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from PIL import Image
# from captum.attr import IntegratedGradients
from scipy.ndimage import gaussian_filter

def predict_and_visualize(
    model,
    grad_cam,
    device,
    image=None,
    url=None,
    class_names=None,
    topk=5,
    alpha=0.3
):
    try:
        # -----------------------------
        # 1. Load image (file OR URL)
        # -----------------------------
        if image is not None:
            if not isinstance(image, Image.Image):
                img_pil = Image.open(image).convert("RGB")
            else:
                img_pil = image
        else:
            try:
                response = requests.get(url)
                response.raise_for_status()
                img_pil = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to load image from URL: {e}")

        # -----------------------------
        # 2. Transforms
        # -----------------------------
        IMAGE_SIZE = 224
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        transform_viz = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_tensor_viz = transform_viz(img_pil).unsqueeze(0).cpu()

        # -----------------------------
        # 3. Predict
        # -----------------------------
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0]

        top_probs, top_idxs = torch.topk(probs, topk)

        # Top-1 prediction
        top1_idx = top_idxs[0].item()
        pred_class = class_names[top1_idx] if class_names else str(top1_idx)
        confidence = top_probs[0].item()

        # -----------------------------    
        # 4. Grad-CAM heatmap (SMOOTHED)
        # -----------------------------
        mask, _ = grad_cam(img_tensor, class_idx=top1_idx)

        mask = gaussian_filter(mask, sigma=5)  # smooth proportional to mask size
        mask = np.clip(mask, 0, 1)

        # Prepare the image
        img_np = img_tensor_viz[0].numpy().transpose(1, 2, 0)  # H x W x C
        img_np = np.clip(img_np, 0, 1)

        # Resize mask to image size
        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Convert mask to heatmap
        heatmap = plt.get_cmap("jet")(mask_resized)[..., :3]  # H x W x 3 float
        superimposed = heatmap * alpha + img_np * (1 - alpha)
        # superimposed = np.clip(superimposed, 0, 1)

        # # Convert to uint8 for PIL
        # superimposed = Image.fromarray(np.uint8(superimposed * 255))

        # After smoothing and blending
        superimposed = np.clip(superimposed, 0, 1)          # still float 0–1
        superimposed = (superimposed * 255).astype(np.uint8)
        superimposed = Image.fromarray(superimposed)  # now safe
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(probs)), probs.cpu().numpy())
        if class_names:
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_title("Xác suất của các lớp")
        plt.tight_layout()
        
        display_size = (512, 512)
        superimposed_resized = superimposed.resize(display_size)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to load image: {e}") 
    return superimposed_resized, pred_class, confidence, fig

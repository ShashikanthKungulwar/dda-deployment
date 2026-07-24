import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import FullModelHAT

CLASS_NAMES = [
    'Background', 'Water', 'Building - No Damage', 'Building - Minor Damage',
    'Building - Major Damage', 'Building - Total Destruction', 'Vehicle',
    'Road - Clear', 'Road - Blocked', 'Tree', 'Pool'
]

CLASS_COLORS = np.array([
    [0,0,0],[61,230,250],[180,120,120],[235,255,7],[255,184,6],
    [255,0,0],[255,0,245],[140,140,140],[160,150,20],[4,250,7],[255,235,0]
], dtype=np.uint8)

device = torch.device("cpu")
model = FullModelHAT(num_classes=11)
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()


def preprocess(image):
    image = image.convert("RGB").resize((384, 384), Image.LANCZOS)
    img_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
    return tensor, image


def build_legend():
    swatch_size = 24
    padding = 8
    row_height = swatch_size + padding
    width = 260
    height = row_height * len(CLASS_NAMES) + padding

    legend = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y = padding + i * row_height
        draw.rectangle(
            [padding, y, padding + swatch_size, y + swatch_size],
            fill=tuple(color.tolist()),
            outline=(0, 0, 0),
        )
        draw.text((padding * 2 + swatch_size, y + swatch_size // 4), name, fill=(0, 0, 0), font=font)

    return legend


LEGEND_IMAGE = build_legend()  # build once, reuse for every prediction


def segment(image, alpha=0.55):
    tensor, resized_image = preprocess(image)
    with torch.no_grad():
        pred_mask, edge_map = model(tensor)
        pred = torch.argmax(pred_mask, dim=1).squeeze(0).numpy()

    color_mask = CLASS_COLORS[pred]
    overlay = (alpha * color_mask + (1 - alpha) * np.array(resized_image)).astype(np.uint8)

    return Image.fromarray(color_mask), Image.fromarray(overlay), LEGEND_IMAGE


demo = gr.Interface(
    fn=segment,
    inputs=[gr.Image(type="pil"), gr.Slider(0, 1, 0.55, label="Overlay opacity")],
    outputs=[
        gr.Image(label="Segmentation Mask"),
        gr.Image(label="Overlay"),
        gr.Image(label="Class Legend"),
    ],
    title="RescueSeg: Aerial Disaster Damage Segmentation",
    description="Hybrid-Attention Transformer trained on RescueNet. 44.0% mIoU (reduced-epoch ablation run).",
)

if __name__ == "__main__":
    demo.launch()
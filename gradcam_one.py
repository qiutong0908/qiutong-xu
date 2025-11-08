import sys, torch, timm, numpy as np, cv2
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from pathlib import Path

IMG_PATH = sys.argv[1]
out_name = "gradcam_" + Path(IMG_PATH).stem + ".jpg"

# load model
ckpt = torch.load("runs_nf_min/best.pt", map_location="cpu")
classes = ckpt["classes"]
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# find last conv
last_conv = None
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        last_conv = m
if last_conv is None:
    raise RuntimeError("No Conv2d found")
print("[Grad-CAM] target layer ->", last_conv.__class__.__name__)

cache = {}
def fwd_hook(_, __, out): cache["feat"] = out.detach()
def bwd_hook(_, grad_in, grad_out): cache["grad"] = grad_out[0].detach()
last_conv.register_forward_hook(fwd_hook)
last_conv.register_full_backward_hook(bwd_hook)

tf = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img_pil = Image.open(IMG_PATH).convert("RGB")
x = tf(img_pil).unsqueeze(0); x.requires_grad_(True)

# forward & backward
logits = model(x)
pred_idx = logits.argmax(1).item()
logits[0, pred_idx].backward()

A = cache["feat"][0]      # [C,H,W]
G = cache["grad"][0]      # [C,H,W]
w = G.mean(dim=(1,2))     # [C]
cam = (w[:,None,None] * A).sum(0).cpu().numpy()
cam = np.maximum(cam, 0)
cam = cam / (cam.max() + 1e-6)

# make sizes match original image
img_np = np.asarray(img_pil)                # (H,W,3)
H, W = img_np.shape[:2]
cam = cv2.resize(cam, (W, H))               # 注意 (W,H)

# overlay
heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
blend = cv2.addWeighted(img_np, 0.35, heat, 0.65, 0)

Image.fromarray(blend).save(out_name)
print(f"Pred: {classes[pred_idx]}  -> saved {out_name}")

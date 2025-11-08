import torch, sys
from torchvision import transforms
from PIL import Image
import timm

ckpt = torch.load("runs_nf_min/best.pt", map_location="cpu")
classes = ckpt["classes"]

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

tf = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img = Image.open(sys.argv[1]).convert("RGB")
x = tf(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
conf, pred = torch.max(prob, dim=0)
print(f"Pred: {classes[pred.item()]}  Conf: {conf.item():.3f}")

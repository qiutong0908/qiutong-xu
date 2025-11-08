# export_torchscript.py
import torch, timm, json

CKPT = "runs_nf_min/best.pt"    
IMG_SIZE = 224                  

ckpt = torch.load(CKPT, map_location="cpu")
classes = ckpt.get("classes", ["NF","nonNF","other"])
with open("classes.json","w") as f:
    json.dump(classes, f)
print("classes:", classes)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
ts = torch.jit.trace(model, dummy)    # or torch.jit.script(model)
ts.save("model.torchscript.pt")
print("OK -> model.torchscript.pt & classes.json")

# predict.py
import argparse, os
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import timm
import csv

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs_nf_min/best.pt")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", ["NF","nonNF","other"])
    num_classes = len(classes)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(torch.device("mps") if device=="mps" else device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    paths=[]
    p=Path(args.input)
    exts={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG",".WEBP",".BMP",".TIF",".TIFF"}
    if p.is_file(): paths=[p]
    else:
        for x in p.rglob("*"):
            if x.suffix in exts: paths.append(x)
    assert paths, "No images found."

    out_csv = Path("preds.csv")
    with open(out_csv,"w",newline="") as f:
        writer = csv.writer(f)
        header = ["path","pred"] + [f"prob_{c}" for c in classes]
        writer.writerow(header)

        for img_path in paths:
            img = Image.open(img_path).convert("RGB")
            x = tf(img).unsqueeze(0)
            if device=="cuda": x=x.cuda(non_blocking=True)
            elif device=="mps": x=x.to(torch.device("mps"))
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(prob.argmax())
            writer.writerow([str(img_path), classes[idx], *[f"{p:.4f}" for p in prob]])

    print(f"Done. CSV saved to {out_csv} with {len(paths)} rows.")

if __name__=="__main__":
    main()

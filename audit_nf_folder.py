import os, csv, argparse, shutil
from PIL import Image
import torch
from torchvision import transforms

# ====== Your model (replace with your own project) ======
# Assume class order: idx 0 = NF, idx 1 = nonNF
from your_model import Net  # Replace with your model definition

def load_model(ckpt):
    model = Net()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def main(args):
    os.makedirs(args.out, exist_ok=True)
    keep_dir = os.path.join(args.out, "NF_confident")   # confidence >= threshold
    drop_dir = os.path.join(args.out, "NF_uncertain")   # confidence < threshold
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(drop_dir, exist_ok=True)

    model = load_model(args.ckpt)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),  # same as during training
        transforms.ToTensor(),
        # If normalization was applied during training, add the same Normalize here
        # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    rows = []
    files = [f for f in os.listdir(args.nf_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    files.sort()
    for f in files:
        p = os.path.join(args.nf_dir, f)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print("bad file:", f, e)
            continue
        x = tfm(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
            p_nf = float(prob[0])
            pred = int(prob.argmax().item())
        rows.append([f, p_nf, pred])

        # Copy to different folders for quick visual inspection
        dst = keep_dir if p_nf >= args.thr else drop_dir
        shutil.copy2(p, os.path.join(dst, f))

    # Export CSV
    with open(os.path.join(args.out, "audit.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "prob_NF", "pred_idx(0=NF,1=nonNF)"])
        w.writerows(rows)

    print("Done. CSV:", os.path.join(args.out, "audit.csv"))
    print("Confident NF:", keep_dir, " | Uncertain/likely non-NF:", drop_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf_dir", required=True, help="Path to NF folder")
    ap.add_argument("--out", default="out_audit", help="Output directory")
    ap.add_argument("--ckpt", default="runs_nf_min/best.pt", help="Path to model checkpoint")
    ap.add_argument("--thr", type=float, default=0.40, help="Confidence threshold for NF")
    args = ap.parse_args()
    main(args)


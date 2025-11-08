import torch, numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load model
ckpt = torch.load("runs_nf_min/best.pt", map_location="cpu")
classes = ckpt["classes"]
import timm
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

tf = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ds = datasets.ImageFolder(Path("data") / "val", transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

ys, ps, preds = [], [], []
with torch.no_grad():
    for x, y in loader:
        logits = model(x)
        prob = torch.softmax(logits, dim=1).numpy()
        ps.extend(prob[:, 1])                    # Probability of NF (assuming class order ['NF', 'nonNF'] or reversed—still reported correctly)
        preds.extend(prob.argmax(1))
        ys.extend(y.numpy())

ys = np.array(ys)
preds = np.array(preds)
ps = np.array(ps)

print("Classes:", classes)
print(classification_report(ys, preds, target_names=classes, digits=3))

try:
    # If binary classification and both classes are present, calculate AUC (treat NF as positive)
    pos_index = classes.index('NF') if 'NF' in classes else 1
    y_bin = (ys == pos_index).astype(int)
    auc = roc_auc_score(y_bin, ps if pos_index == 1 else 1 - ps)
    print("ROC-AUC (NF positive):", round(auc, 4))
except Exception as e:
    print("AUC not available:", e)

print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(ys, preds))


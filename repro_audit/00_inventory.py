"""Read-only inventory: hashes, checkpoint metadata, h5 structure. Writes repro_audit/00_inventory.txt"""
import hashlib, json, os, subprocess, sys, time
import h5py, numpy as np, torch

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "repro_audit", "00_inventory.txt")
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

P("git commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
files = ["models/model.pth", "models/model.onnx", "models/model_int8.onnx", "assets/dataset.h5"]
for p in files:
    if os.path.exists(p):
        P(f"{p}: {os.path.getsize(p)} bytes, mtime {time.ctime(os.path.getmtime(p))}, sha256 {sha(p)}")
    else:
        P(f"{p}: MISSING")

ck = torch.load("models/model.pth", map_location="cpu", weights_only=False)
P("\ncheckpoint keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
for k in ("epoch", "val_acc", "num_classes"):
    P(f"  {k}:", ck.get(k))
cls = ck.get("classes") or []
P("  len(classes):", len(cls), "| has __reject__:", "__reject__" in cls)
sd = ck.get("model_state_dict", ck)
dom = [k for k in sd if "domain" in k.lower()]
P("  domain-head params in checkpoint:", dom[:5])

with h5py.File("assets/dataset.h5", "r") as f:
    P("\nh5 attrs:", {k: (v if not isinstance(v, bytes) else v.decode()) for k, v in f.attrs.items()})
    for k in f.keys():
        d = f[k]
        P(f"  /{k}: shape={getattr(d,'shape',None)} dtype={getattr(d,'dtype',None)}")
    labels = f["labels"][:]
    P("  n samples:", len(labels), "classes present:", len(np.unique(labels)))
    names = json.loads(f["class_names"][()])
    P("  h5 class_names count:", len(names))
    if "domain_names" in f:
        P("  domains:", json.loads(f["domain_names"][()]))
        d, c = np.unique(f["domains"][:], return_counts=True)
        P("  domain counts:", dict(zip(d.tolist(), c.tolist())))
open(OUT, "w", encoding="utf-8").write("\n".join(lines))

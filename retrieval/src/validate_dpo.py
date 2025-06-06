# validate_retrieval_dpo_nooti.py
"""
Validation script for Retrieval‑DPO on FashionIQ.
Computes Recall@1/5/10/50 over the val split.
"""

import argparse
import json
import random
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
import clip
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

# ─── Preprocessing ──────────────────────────────────────────────────────────
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466,0.4578275,0.40821073),
              (0.26862954,0.26130258,0.27577711)),
])

# ─── FAISS for gallery (reuse from train) ───────────────────────────────────
class FaissIndex:
    def __init__(self, img_dir: Path, device: torch.device, cache_path: Path = None):
        self.device = device
        self.img_dir = Path(img_dir)
        self.cache_path = Path(cache_path) if cache_path else None
        self.row2fname = {}
        self._build_or_load_index()

    def _build_or_load_index(self):
        if self.cache_path and self.cache_path.exists():
            idx = faiss.read_index(str(self.cache_path))
            with open(self.cache_path.with_suffix('.pkl'), 'rb') as f:
                self.row2fname = pickle.load(f)
            self.index = idx
            return

        imgs = sorted(self.img_dir.glob("*.jpg"))
        self.row2fname = {i: f.name for i, f in enumerate(imgs)}

        model, _ = clip.load("ViT-L/14", device=self.device, jit=False)
        model = model.float().eval()

        feats = []
        for img_path in tqdm(imgs, desc="encode gallery"):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad(), autocast():
                feats.append(model.encode_image(img).cpu().numpy())
        feats = np.vstack(feats).astype("float32")
        faiss.normalize_L2(feats)

        idx = faiss.IndexFlatIP(feats.shape[1])
        idx.add(feats)
        self.index = idx

        if self.cache_path:
            faiss.write_index(idx, str(self.cache_path))
            with open(self.cache_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.row2fname, f)

# ─── Validation dataset (no OTI) ────────────────────────────────────────────
class FashionIQValNoOTIDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.queries = []
        cats = ["dress","shirt","toptee"]
        for c in cats:
            capf = self.root/"captions"/f"cap.{c}.{split}.json"
            splitf = self.root/"image_splits"/f"split.{c}.{split}.json"
            capd = json.load(open(capf))
            sids = set(json.load(open(splitf)))
            for e in capd:
                if e["target"] in sids and e["candidate"] in sids:
                    for cap in e["captions"]:
                        # store (caption, target_filename)
                        self.queries.append((cap, e["candidate"] + ".jpg"))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        cap, fname = self.queries[idx]
        txt = clip.tokenize(f"<|image|> {cap}")[0]
        return txt, fname

# ─── Recall@K computation ──────────────────────────────────────────────────
@torch.no_grad()
def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load CLIP + fine-tuned transformer
    model, _ = clip.load("ViT-L/14", device=device, jit=False)
    model = model.float().eval()
    state = torch.load(args.weights, map_location=device)
    model.transformer.load_state_dict(state)
    for p in model.visual.parameters():
        p.requires_grad_(False)

    # build gallery index
    gallery = FaissIndex(Path(args.dataset)/"images", device, cache_path=args.cache_index)
    index = gallery.index

    # load val data
    ds = FashionIQValNoOTIDataset(args.dataset, "val")
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)

    total = rec1 = rec5 = rec10 = rec50 = 0

    for txt, fnames in tqdm(dl, desc="validating"):
        txt = txt.to(device)
        with autocast():
            emb = model.token_embedding(txt)
            x = emb + model.positional_embedding
            x = model.transformer(x.permute(1,0,2)).permute(1,0,2)
            x = model.ln_final(x)
            feats = x[torch.arange(x.size(0)), txt.argmax(-1)]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype("float32")

        D, I = index.search(feats, 50)

        for row_idx, fname in enumerate(fnames):
            total += 1
            # find target row
            # (we know gallery.row2fname maps row→filename)
            # so invert it once:
            name2row = {v:k for k,v in gallery.row2fname.items()}
            tgt_row = name2row[fname]
            nbrs = I[row_idx]

            if tgt_row in nbrs[:1]:
                rec1 += 1; rec5 += 1; rec10 += 1; rec50 += 1
            elif tgt_row in nbrs[:5]:
                rec5 += 1; rec10 += 1; rec50 += 1
            elif tgt_row in nbrs[:10]:
                rec10 += 1; rec50 += 1
            elif tgt_row in nbrs[:50]:
                rec50 += 1

    print(f"Recall@1  : {rec1/total*100:.2f}")
    print(f"Recall@5  : {rec5/total*100:.2f}")
    print(f"Recall@10 : {rec10/total*100:.2f}")
    print(f"Recall@50 : {rec50/total*100:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True, help="path to datasets/FashionIQ")
    p.add_argument("--weights",    required=True, help="checkpoint .pt file")
    p.add_argument("--batch",      type=int, default=256)
    p.add_argument("--workers",    type=int, default=6)
    p.add_argument("--cache-index", help="optional path to FAISS index .index/.pkl")
    args = p.parse_args()
    validate(args)

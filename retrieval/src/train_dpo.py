# train_retrieval_dpo_nooti.py
"""
Retrieval‑DPO for FashionIQ, with FAISS caching
and mixed‑precision.
"""
import argparse
import random
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
import clip
from PIL import Image
from torch import nn
from torch.cuda.amp import autocast, GradScaler
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

def dpo_loss(sim_p, sim_n, beta):
    return -torch.log(torch.sigmoid(beta * (sim_p - sim_n))).mean()

# ─── FAISS for hard negatives w/ optional caching ───────────────────────────
class FaissIndex:
    def __init__(self, img_dir: Path, device: torch.device, cache_path: Path = None):
        self.device = device
        self.img_dir = Path(img_dir)
        self.cache_path = Path(cache_path) if cache_path else None
        self.row2fname = {}
        self._build_or_load_index()

    def _build_or_load_index(self):
        # look for cache
        if self.cache_path and self.cache_path.exists():
            idx = faiss.read_index(str(self.cache_path))
            with open(self.cache_path.with_suffix('.pkl'), 'rb') as f:
                self.row2fname = pickle.load(f)
            self.index = idx
            return

        # otherwise build from scratch
        imgs = sorted(self.img_dir.glob("*.jpg"))
        self.row2fname = {i: f.name for i, f in enumerate(imgs)}

        model, _ = clip.load("ViT-L/14", device=self.device, jit=False)
        model = model.float().eval()

        feats = []
        for img_path in tqdm(imgs, desc="encode gallery"):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with autocast():
                    v = model.encode_image(img)
                feats.append(v.cpu().numpy())
        feats = np.vstack(feats).astype("float32")
        faiss.normalize_L2(feats)

        idx = faiss.IndexFlatIP(feats.shape[1])
        idx.add(feats)

        self.index = idx

        # save cache
        if self.cache_path:
            faiss.write_index(idx, str(self.cache_path))
            with open(self.cache_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.row2fname, f)

    def hard_negative(self, pos_row: int, k: int = 50) -> int:
        vec = self.index.reconstruct(pos_row)[None]
        _, I = self.index.search(vec, k + 1)
        for nbr in I[0]:
            if nbr != pos_row:
                return int(nbr)
        return random.randrange(self.index.ntotal)

# ─── Dataset WITHOUT OTI ────────────────────────────────────────────────────
class FashionIQNoOTIDataset(Dataset):
    def __init__(self, root: Path, split: str, faiss_idx: FaissIndex):
        self.root = Path(root)
        self.faiss = faiss_idx
        self.queries = []
        cats = ["dress","shirt","toptee"]
        for c in cats:
            caps = json.load(open(self.root/"captions"/f"cap.{c}.{split}.json"))
            sids = set(json.load(open(self.root/"image_splits"/f"split.{c}.{split}.json")))
            for e in caps:
                if e["target"] in sids and e["candidate"] in sids:
                    for cap in e["captions"]:
                        self.queries.append((cap, e["candidate"] + ".jpg"))
        # map filename→row
        self.name2row = self.faiss.row2fname

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        cap, tgt_name = self.queries[i]
        txt = clip.tokenize(f"<|image|> {cap}")[0]
        pos = preprocess(Image.open(self.root/"images"/tgt_name).convert("RGB"))
        row_t = {v:k for k,v in self.faiss.row2fname.items()}[tgt_name]
        neg_row = self.faiss.hard_negative(row_t)
        neg_name = self.faiss.row2fname[neg_row]
        neg = preprocess(Image.open(self.root/"images"/neg_name).convert("RGB"))
        return txt, pos, neg

# ─── Training ────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load("ViT-L/14", device=device, jit=False)
    model = model.float().eval()
    for p in model.visual.parameters():
        p.requires_grad_(False)

    # prepare FAISS
    fa = FaissIndex(args.dataset + "/images", device, cache_path=args.cache_index)

    # data
    ds = FashionIQNoOTIDataset(args.dataset, "train", fa)
    dl = DataLoader(ds, args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

    # optimizer & scaler
    opt = torch.optim.AdamW(model.transformer.parameters(),
                             lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler()

    for ep in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {ep}", unit="batch")
        for txt, pos, neg in pbar:
            txt, pos, neg = txt.to(device), pos.to(device), neg.to(device)
            with autocast():
                # text forward
                emb = model.token_embedding(txt)
                x = emb + model.positional_embedding
                x = model.transformer(x.permute(1,0,2)).permute(1,0,2)
                x = model.ln_final(x)
                tfeat = x[torch.arange(x.size(0)), txt.argmax(-1)]
                tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

                # image forward
                pf = model.encode_image(pos); pf = pf / pf.norm(dim=-1, keepdim=True)
                nf = model.encode_image(neg); nf = nf / nf.norm(dim=-1, keepdim=True)

                sim_p = (tfeat * pf).sum(-1) * model.logit_scale.exp()
                sim_n = (tfeat * nf).sum(-1) * model.logit_scale.exp()
                loss = dpo_loss(sim_p, sim_n, args.beta)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=loss.item())

        # checkpoint
        out_path = Path(args.out)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.transformer.state_dict(),
                   out_path / f"epoch{ep}.pt")

if __name__ == "__main__":
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="path to datasets/FashionIQ")
    p.add_argument("--out", required=True,
                   help="output folder for checkpoints")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cache-index", required=False,
                   help="path to save/load FAISS index")
    args = p.parse_args()
    train(args)

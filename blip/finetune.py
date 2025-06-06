#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime
from statistics import mean
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lavis.models import load_model_and_preprocess

# ---------------------------------------------------------------------------
# Assume combiner.py exports MoECombiner
# ---------------------------------------------------------------------------
from combiner import MoECombiner

# ---------------------------------------------------------------------------
# Minimal FashionIQ dataset stub (swap in your real one)
# ---------------------------------------------------------------------------
class FashionIQDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess, root: Path):
        self.paths: List[Path] = []
        img_dir = root / "images"
        for dt in dress_types:
            # you can refine the glob pattern as needed
            for ext in ("png", "jpg", "jpeg"):
                self.paths += list(img_dir.glob(f"*{dt}*.{ext}"))
        self.pre = preprocess
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.pre(img)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def squarepad_transform(sz: int):
    import torchvision.transforms as T
    return T.Compose([T.Resize((sz, sz)), T.ToTensor()])

def targetpad_transform(r: float, sz: int):
    import torchvision.transforms as T
    pad = int(sz*(1-r)/2)
    return T.Compose([T.Resize((int(sz*r), sz)), T.Pad(pad), T.ToTensor()])

# ---------------------------------------------------------------------------
# Helpers & stubs (swap in your real implementations)
# ---------------------------------------------------------------------------
def collate_fn(batch):
    # in pipeline we just need image tensors
    return torch.stack(batch), torch.stack(batch), [""]*len(batch)

def save_model(name, epoch, model, path: Path):
    torch.save({"epoch": epoch, "state": model.state_dict()}, path / f"{name}.pt")

def build_dataloaders(root: Path, train_types, val_types, transform, batch, workers, ratio):
    pre = squarepad_transform(224) if transform=="squarepad" else targetpad_transform(ratio,224)
    train = DataLoader(
        FashionIQDataset("train", train_types, "relative", pre, root),
        batch_size=batch, shuffle=True, drop_last=True,
        num_workers=workers, collate_fn=collate_fn
    )
    rel_val, cls_val = [], []
    for t in val_types:
        rel_val.append(FashionIQDataset("val", [t], "relative", pre, root))
        cls_val.append(FashionIQDataset("val", [t], "classic",  pre, root))
    return train, rel_val, cls_val

# these should be your real utilities:
from utils import (
    generate_randomized_fiq_caption,
    update_train_running_results_dict,
    set_train_bar_description_dict,
    extract_index_blip_features,
)
from validate_blip import compute_fiq_val_metrics

# ---------------------------------------------------------------------------
# Zero-shot pipeline (BLIP-2 + MoECombiner + cosine)
# ---------------------------------------------------------------------------
def run_pipeline(root: Path, topk: int, alpha: float, out_file: str, device: str):
    # 1) load BLIP-2
    model, V, T = load_model_and_preprocess(
        "blip2_feature_extractor", "pretrain", is_eval=True, device=device
    )

    # 2) infer embed dim at runtime
    dummy = torch.zeros(1,3,224,224, device=device)
    with torch.no_grad():
        D = model.encode_image(dummy).shape[-1]
    combiner = MoECombiner(D).to(device).eval()

    # 3) build gallery embeddings
    gallery = FashionIQDataset("all", [""], "classic", V["eval"], root)
    gloader = DataLoader(gallery, batch_size=16, num_workers=4, collate_fn=lambda b: torch.stack(b))
    emb = []
    with torch.no_grad():
        for imgs in tqdm(gloader, desc="gallery enc"):
            imgs = imgs.to(device)
            out = model({"image": imgs, "text_input": None})["image_embeds"]
            emb.append(out.float())
    gallery_emb = F.normalize(torch.cat(emb), dim=-1)

    # 4) dummy queries = first 100
    results = {}
    for qi in tqdm(range(min(100, len(gallery))), desc="queries"):
        ref = gallery_emb[qi:qi+1]
        # mix with zero-text via combiner
        query = combiner(ref, ref)  # e.g. expert mix gates on image alone
        sims = (query @ gallery_emb.T).squeeze(0)
        topk_idx = sims.topk(topk).indices.cpu().tolist()
        results[str(gallery.paths[qi])] = [str(gallery.paths[i]) for i in topk_idx]

    # 5) dump
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"✅ Wrote {out_file} with {len(results)} queries")

# ---------------------------------------------------------------------------
# Fine-tuning trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, root: Path, tr_types, v_types, epochs, batch, lr, workers, transform, ratio, device):
        self.dev = device
        # 1) load BLIP-2 trainable
        self.model, V, T = load_model_and_preprocess(
            "blip2_feature_extractor", "pretrain", is_eval=False, device=device
        )
        if callable(getattr(self.model, "_update_f_former", None)):
            self.model._update_f_former()

        # 2) infer D and build MoE head
        dummy = torch.zeros(1,3,224,224, device=device)
        with torch.no_grad():
            D = self.model.encode_image(dummy).shape[-1]
        self.combiner = MoECombiner(D).to(device)

        # 3) dataloaders
        self.train_loader, self.rel_val, self.cls_val = build_dataloaders(
            root, tr_types, v_types, transform, batch, workers, ratio
        )

        # 4) optimizer & scheduler
        params = filter(lambda p: p.requires_grad, list(self.model.parameters())+list(self.combiner.parameters()))
        self.opt = optim.AdamW(params, lr=lr, betas=(0.9,0.98), eps=1e-7, weight_decay=0.05)
        self.sched = OneCycleLR(self.opt, max_lr=lr, pct_start=1.5/epochs,
                                div_factor=100, steps_per_epoch=len(self.train_loader), epochs=epochs)
        self.scaler = torch.cuda.amp.GradScaler()
        self.epochs = epochs

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ckpt_dir = Path("models")/f"fiq_{ts}"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best = 0.

    def train(self):
        for ep in range(self.epochs):
            self.model.train(); self.combiner.train()
            run = {"images":0, "loss_itc":0., "loss_rtc":0., "loss_align":0.}
            pbar = tqdm(self.train_loader, desc=f"Ep {ep+1}/{self.epochs}")
            for ref, tgt, caps in pbar:
                ref, tgt = ref.to(self.dev), tgt.to(self.dev)
                rnd = generate_randomized_fiq_caption(np.array(caps).T.flatten().tolist())
                txt_in = [T["eval"](c) for c in rnd]

                self.opt.zero_grad()
                with torch.cuda.amp.autocast():
                    losses = self.model({"image":ref, "target":tgt, "text_input":txt_in})
                    # optionally combine image+text feats too:
                    # img_emb = self.model.encode_image(ref)
                    # text_emb = self.model.encode_text(self.model.tokenizer(rnd).to(self.dev))
                    # combined = self.combiner(img_emb, text_emb)
                    loss = sum(losses.values())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()

                update_train_running_results_dict(run, losses, ref.size(0))
                pbar.set_postfix({
                    k: (run[k]/run["images"]) for k in run if k!="images"
                })

            # checkpoint
            self.validate(ep)

    def validate(self, epoch):
        self.model.eval(); self.combiner.eval()
        rec10, rec50 = [], []
        for rel_ds, cls_ds in zip(self.rel_val, self.cls_val):
            idx_feats, idx_names = extract_index_blip_features(cls_ds, self.model, save_memory=False)
            r10, r50 = compute_fiq_val_metrics(rel_ds, self.model, idx_feats, idx_names, None, save_memory=False)
            rec10.append(r10); rec50.append(r50)

        avg = (mean(rec10)+mean(rec50))/2
        print(json.dumps({
            "epoch":epoch,
            "avg_r10": mean(rec10),
            "avg_r50": mean(rec50),
            "avg_rec": avg
        }, indent=2))
        if avg>self.best:
            self.best=avg
            save_model("best", epoch, self.model, self.ckpt_dir)

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode", required=True)

    pipe = sp.add_parser("pipeline")
    pipe.add_argument("--root",   required=True, help="FashionIQ root")
    pipe.add_argument("--topk",   type=int, default=50)
    pipe.add_argument("--alpha",  type=float, default=0.5)
    pipe.add_argument("--out",    required=True, help="output json path")

    fin = sp.add_parser("finetune")
    fin.add_argument("--root",        required=True, help="FashionIQ root")
    fin.add_argument("--train_types", nargs="+", required=True)
    fin.add_argument("--val_types",   nargs="+", required=True)
    fin.add_argument("--epochs",      type=int, default=5)
    fin.add_argument("--batch",       type=int, default=32)
    fin.add_argument("--lr",          type=float, default=2e-5)
    fin.add_argument("--num_workers", type=int, default=8)
    fin.add_argument("--transform",   choices=["squarepad","targetpad"], default="squarepad")
    fin.add_argument("--target_ratio",type=float, default=0.9)

    return p.parse_args()

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    dev  = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode=="pipeline":
        run_pipeline(Path(args.root), args.topk, args.alpha, args.out, dev)
    else:
        tr = Trainer(
            root=Path(args.root),
            tr_types=args.train_types,
            v_types=args.val_types,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            workers=args.num_workers,
            transform=args.transform,
            ratio=args.target_ratio,
            device=dev
        )
        tr.train()

if __name__=="__main__":
    main()

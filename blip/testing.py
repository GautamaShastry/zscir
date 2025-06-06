#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
from PIL import Image
from lavis.models import load_model_and_preprocess

from data_utils import squarepad_transform, targetpad_transform
from utils import device

def load_checkpoint(model, ckpt_path: Path):
    """
    Inject only the keys your checkpoint provides (e.g. prompt tokens, heads)
    into the already-initialized BLIP-2 model, ignoring any missing keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # drill down if nested under the class name
    if isinstance(ckpt, dict):
        cls_name = model.__class__.__name__
        if cls_name in ckpt and isinstance(ckpt[cls_name], dict):
            state_dict = ckpt[cls_name]
            print(f"ℹ️  Using ckpt['{cls_name}'] as state_dict")
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded {len(state_dict)} keys from {ckpt_path.name}")
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    return model

def get_preprocessor(transform: str, target_ratio: float = 1.25, input_dim: int = 224):
    if transform == "squarepad":
        return squarepad_transform(input_dim)
    elif transform == "targetpad":
        return targetpad_transform(target_ratio, input_dim)
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Test a single example through a fine-tuned BLIP-2/CLIP model"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to your saved model checkpoint (.pt/.pth)")
    parser.add_argument("--ref-image", type=Path, required=True,
                        help="path to the reference image")
    parser.add_argument("--target-image", type=Path,
                        help="path to the ground-truth target image (if available)")
    parser.add_argument("--caption", type=str, required=True,
                        help="the relative text describing the change")
    parser.add_argument("--model-name", type=str, default="blip2_cir_cat",
                        help="BLIP-2 name (e.g. blip2_cir_cat, blip2_cir_align_prompt)")
    parser.add_argument("--backbone", type=str, default="pretrain",
                        help="backbone for BLIP-2 (e.g. pretrain, pretrain_vitL)")
    parser.add_argument("--transform", type=str, default="squarepad",
                        choices=["squarepad", "targetpad"],
                        help="preprocessing pipeline")
    parser.add_argument("--target-ratio", type=float, default=1.25,
                        help="only used if --transform targetpad")
    args = parser.parse_args()

    # 1) Load model & preprocessors
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.backbone,
        is_eval=not bool(args.target_image),
        device=device
    )
    model = load_checkpoint(model, args.checkpoint)
    model.eval()

    # 2) Build preprocessing pipeline
    custom_preproc = get_preprocessor(args.transform, args.target_ratio)
    def prep(img_path: Path):
        img = Image.open(img_path).convert("RGB")
        return custom_preproc(img) if custom_preproc else vis_processors["eval"](img)

    # 3) Preprocess your single example
    ref_tensor = prep(args.ref_image).unsqueeze(0).to(device)
    if args.target_image:
        target_tensor = prep(args.target_image).unsqueeze(0).to(device)
    text_tokens = txt_processors["eval"](args.caption)

    # 4) Run forward pass & print results
    with torch.no_grad():
        if args.target_image:
            out = model({
                "image": ref_tensor,
                "target": target_tensor,
                "text_input": [text_tokens]
            })
            print("\n=== Loss components for this example ===")
            for key, value in out.items():
                print(f"{key:20s}: {value.item():.4f}")
        else:
            feats = model({"image": ref_tensor, "text_input": [text_tokens]})
            print("\n=== Inference output ===")
            print(feats)

if __name__ == "__main__":
    main()

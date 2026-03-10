"""
SAM3 Mask Extractor for LichtFeld Studio / Gaussian Splatting
=============================================================
Generates per-group masks (one folder per subject)

Requirements:
    pip install ultralytics>=8.3.237
    pip uninstall clip -y
    pip install git+https://github.com/ultralytics/CLIP.git

    Re-install torch version according to CUDA arch, ignore if running CPU

    SAM3 weights: request access at https://huggingface.co/facebook/sam3


Usage:
    python extract_masks_sam3.py \
        --image ./images/ \
        --group plants "crop plants" \
        --group persons "person" "human" \
        --fill-holes \
        --dilate 8 \
        --out ./masks


Output:
    masks/
    ├── plants/          ← plants only — LichtFeld training run 1
    │   ├── img_0.jpg.png
    │   └── ...
    └── persons/         ← persons only — LichtFeld training run 2
        ├── img_0.jpg.png
        └── ...
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download


def ensure_sam3(checkpoint: str = "sam3.pt"):
    try:
        import ultralytics
        from packaging import version
        if version.parse(ultralytics.__version__) < version.parse("8.3.237"):
            sys.exit(
                f"Ultralytics {ultralytics.__version__} is too old.\n"
                "Run: pip install -U ultralytics"
            )
    except ImportError:
        sys.exit("Run: pip install ultralytics>=8.3.237")
    if not Path(checkpoint).exists():
        hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir=str(Path(checkpoint).parent))


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    combined = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        combined |= m.astype(bool)
    return combined


def save_mask(mask: np.ndarray, out_path: Path):
    result = (mask.astype(np.uint8)) * 255
    Image.fromarray(result, mode="L").save(out_path)


# ── SAM3 inference ────────────────────────────────────────────────────────────

def run_text_mode(predictor, image_path: str, prompts: list[str], orig_size: tuple[int, int]) -> list[np.ndarray]:
    orig_w, orig_h = orig_size
    predictor.set_image(image_path)
    results = predictor(text=prompts)

    masks = []
    for result in results:
        if result.masks is None:
            continue
        for mask_tensor in result.masks.data:
            mask = mask_tensor.cpu().numpy().astype(np.uint8)
            mask = np.array(Image.fromarray(mask).resize((orig_w, orig_h), Image.NEAREST))
            masks.append(mask.astype(bool))
    return masks


# ── Per-image processing ──────────────────────────────────────────────────────

def process_image(image_path: Path, groups: list[tuple[str, list[str]]],
                  out_dir: Path, predictor, fill_holes: bool = False, dilate: int = 0):
    print(f"\n[SAM3] {image_path.name}")
    filename = image_path.name + image_path.suffix  # COLMAP convention: img.jpg → img.jpg.jpg

    img = Image.open(image_path)
    orig_size = img.size  # (width, height)
    group_masks: dict[str, np.ndarray] = {}

    # Pass 1: run inference for all groups
    for group_name, prompts in groups:
        print(f"    [{group_name}] prompts: {prompts}")
        masks = run_text_mode(predictor, str(image_path), prompts, orig_size)

        if not masks:
            print(f"    [{group_name}] no masks found — writing empty mask")
            blank = np.zeros((img.height, img.width), dtype=bool)
            group_masks[group_name] = blank
        else:
            print(f"    [{group_name}] {len(masks)} mask(s), merging...")
            group_masks[group_name] = union_masks(masks)

    # Pass 2: postprocess, subtract other groups, save
    if fill_holes or dilate:
        from mask_utils import postprocess
    group_names = list(group_masks.keys())
    for group_name in group_names:
        m = group_masks[group_name]
        if fill_holes or dilate:
            m = postprocess(m, fill_holes, dilate)
        # Remove any pixels claimed by earlier (higher-priority) groups
        for other in group_names:
            if other == group_name:
                break
            m = m & ~group_masks[other]
        group_masks[group_name] = m
        save_mask(m, out_dir / group_name / filename)


# ── CLI ───────────────────────────────────────────────────────────────────────

class GroupAction(argparse.Action):
    """Collect --group name prompt1 prompt2 ... into a list of (name, [prompts])."""
    def __call__(self, parser, namespace, values, option_string=None):
        groups = getattr(namespace, self.dest, None) or []
        name = values[0]
        prompts = values[1:]
        if not prompts:
            parser.error(f"--group {name} needs at least one prompt")
        groups.append((name, prompts))
        setattr(namespace, self.dest, groups)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 mask extractor"
    )
    parser.add_argument("--image", required=True,
                        help="Image file or folder of images")
    parser.add_argument("--group", dest="groups", action=GroupAction,
                        nargs="+", metavar=("NAME", "PROMPT"),
                        help="Named group: --group plants 'crop plants' 'leaves' 'stems'")
    parser.add_argument("--checkpoint", default="sam3.pt",
                        help="Path to sam3.pt (default: sam3.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--out", default="masks",
                        help="Output root directory (default: ./masks)")
    parser.add_argument("--fill-holes", action="store_true",
                        help="Fill enclosed holes in each mask (e.g. rocks inside soil)")
    parser.add_argument("--dilate", type=int, default=0, metavar="N",
                        help="Morphological closing radius in pixels to bridge gaps (default: off)")
    args = parser.parse_args()

    if not args.groups:
        parser.error("Provide at least one --group, e.g.:\n"
                     "  --group plants 'crop plants' 'leaves'\n"
                     "  --group persons 'person' 'human'")

    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[device] Using Apple MPS")
    else:
        device = "cpu"
        print("[device] WARNING: No GPU found, using CPU")
    user_approval = input("Continue? (y/n) ")
    if user_approval != "y":
        sys.exit()

    ensure_sam3(args.checkpoint)

    _start = time.time()
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = dict(conf=args.conf, task="segment", mode="predict", model=args.checkpoint, device=device, save=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)

    out_dir = Path(args.out)
    for name, _ in args.groups:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if image_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = [p for p in sorted(image_path.iterdir()) if p.suffix.lower() in exts]
        if not images:
            sys.exit(f"No images found in {image_path}")
        for img in images:
            process_image(img, args.groups, out_dir, predictor, args.fill_holes, args.dilate)
    elif image_path.is_file():
        process_image(image_path, args.groups, out_dir, predictor, args.fill_holes, args.dilate)
    else:
        sys.exit(f"Path not found: {args.image}")

    print(f"\nDone in {time.time() - _start:.1f} seconds!")
    print(f"    Masks saved to:\n    {out_dir.resolve()}")

if __name__ == "__main__":
    main()

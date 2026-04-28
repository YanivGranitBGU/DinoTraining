import os
import argparse

import torch

import utils
import vision_transformer as vits


def build_models(arch: str, patch_size: int, lora_rank: int, lora_alpha: float):
    if arch not in vits.__dict__:
        raise ValueError(f"Unknown ViT architecture '{arch}'. Available: {list(vits.__dict__.keys())}")

    # Base ViT without LoRA
    base_model = vits.__dict__[arch](patch_size=patch_size)

    # LoRA-enabled ViT (same backbone, attention linears wrapped with LoRA)
    lora_model = vits.__dict__[arch](
        patch_size=patch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    return base_model, lora_model


def main():
    parser = argparse.ArgumentParser("LoRA equivalence test: base ViT vs LoRA-wrapped ViT")
    parser.add_argument("--arch", default="vit_base", type=str, help="ViT architecture (vit_tiny | vit_small | vit_base)")
    parser.add_argument("--patch_size", default=8, type=int, help="Patch size of the ViT")
    parser.add_argument("--pretrained_weights", default="../dino_vitbase8_pretrain_full_checkpoint.pth", type=str,
                        help="Path to the DINO checkpoint (non-LoRA) to load into both models")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help="Checkpoint key inside the .pth file (e.g. teacher | student)")
    parser.add_argument("--lora_rank", default=8, type=int, help="LoRA rank r for the attention projections")
    parser.add_argument("--lora_alpha", default=16.0, type=float, help="LoRA scaling alpha")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size of random inputs for the test")
    parser.add_argument("--height", default=224, type=int, help="Input image height")
    parser.add_argument("--width", default=224, type=int, help="Input image width")
    parser.add_argument("--atol", default=1e-5, type=float, help="Absolute tolerance for allclose check")
    parser.add_argument("--rtol", default=1e-5, type=float, help="Relative tolerance for allclose check")

    args = parser.parse_args()

    if not os.path.isfile(args.pretrained_weights):
        raise FileNotFoundError(f"Pretrained weights not found at: {args.pretrained_weights}")

    print("Building base and LoRA models...")
    base_model, lora_model = build_models(
        arch=args.arch,
        patch_size=args.patch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # Load the same pretrained (non-LoRA) DINO checkpoint into both models
    # For the base model we use the standard loader, for the LoRA model we
    # use the LoRA-aware loader that remaps attention projection keys into
    # the inner linear modules of LoRALinear.
    print("Loading pretrained weights into base model...")
    utils.load_pretrained_weights(
        base_model,
        pretrained_weights=args.pretrained_weights,
        checkpoint_key=args.checkpoint_key,
        model_name=args.arch,
        patch_size=args.patch_size,
    )

    print("Loading pretrained weights into LoRA model (base weights only)...")
    utils.load_pretrained_weights_for_lora(
        lora_model,
        pretrained_weights=args.pretrained_weights,
        checkpoint_key=args.checkpoint_key,
        model_name=args.arch,
        patch_size=args.patch_size,
    )

    base_model.eval()
    lora_model.eval()

    # Random but fixed input so the comparison is deterministic
    torch.manual_seed(0)
    x = torch.randn(args.batch_size, 3, args.height, args.width)

    with torch.no_grad():
        base_out = base_model(x)
        lora_out = lora_model(x)

    diff = (base_out - lora_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("Base output shape:", tuple(base_out.shape))
    print("LoRA output shape:", tuple(lora_out.shape))
    print(f"Max |base - lora|:  {max_diff:.6e}")
    print(f"Mean |base - lora|: {mean_diff:.6e}")

    if torch.allclose(base_out, lora_out, atol=args.atol, rtol=args.rtol):
        print("[OK] LoRA model matches base model when LoRA params are untrained (only base weights loaded).")
    else:
        raise AssertionError(
            f"[FAIL] LoRA model output differs from base model beyond tolerance. "
            f"max_diff={max_diff:.6e}, atol={args.atol}, rtol={args.rtol}"
        )


if __name__ == "__main__":
    main()

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

import utils
import vision_transformer as vits


def _dataset_split_meta(pt_path: str) -> Tuple[int, int]:
    data = torch.load(pt_path, map_location="cpu")
    if "samples" not in data:
        raise ValueError(f"Unsupported dataset format in {pt_path}: missing 'samples'")
    samples = data["samples"]
    if samples.ndim != 3:
        raise ValueError(f"Expected samples shape [N,C,L] in {pt_path}, got {tuple(samples.shape)}")
    n_from_samples = int(samples.shape[0])
    seq_len = int(samples.shape[2])

    if "labels" in data:
        n = int(data["labels"].shape[0])
        # Use samples count if there is any mismatch for robustness.
        if n != n_from_samples:
            n = n_from_samples
    else:
        n = n_from_samples
    return n, seq_len


def _dataset_has_nan_or_inf(pt_path: str) -> Tuple[bool, bool]:
    data = torch.load(pt_path, map_location="cpu")
    if "samples" not in data:
        raise ValueError(f"Unsupported dataset format in {pt_path}: missing 'samples'")
    samples = data["samples"].float()
    has_nan = bool(torch.isnan(samples).any().item())
    has_inf = bool(torch.isinf(samples).any().item())
    return has_nan, has_inf


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def _sanitize_tag(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)


def _result_filename(args) -> str:
    if args.result_filename:
        return args.result_filename
    mode = "linearfinetune" if args.linear_finetune else ("zeroshot_bestperm" if args.search_label_permutations else "zeroshot")
    dataset_tag = "all" if args.dataset.lower() == "all" else _sanitize_tag(args.dataset)
    checkpoint_tag = _sanitize_tag(args.checkpoint_tag) if args.checkpoint_tag else "checkpoint"
    return f"eval_ucr_clsa_results_{dataset_tag}_{mode}_{checkpoint_tag}.json"


class UCRCLSAImageDataset(Dataset):
    """Dataset for UCR_CLSA train.pt/test.pt files.

    Expected file format:
      {"samples": Tensor[N, C, L], "labels": Tensor[N]}
    """

    def __init__(
        self,
        pt_path: str,
        transform=None,
        channel_mins: Optional[torch.Tensor] = None,
        channel_maxs: Optional[torch.Tensor] = None,
    ) -> None:
        data = torch.load(pt_path, map_location="cpu")
        self.samples = data["samples"].float()  # [N, C, L]
        self.labels = data["labels"].long()
        self.transform = transform

        if self.samples.ndim != 3:
            raise ValueError(f"Expected samples shape [N, C, L], got {tuple(self.samples.shape)}")

        if channel_mins is None or channel_maxs is None:
            mins = self.samples.amin(dim=(0, 2))
            maxs = self.samples.amax(dim=(0, 2))
            self.channel_mins = mins
            self.channel_maxs = maxs
        else:
            self.channel_mins = channel_mins
            self.channel_maxs = channel_maxs

    def __len__(self) -> int:
        return self.samples.shape[0]

    def _to_pil(self, x: torch.Tensor) -> Image.Image:
        # x: [C, L]
        mins = self.channel_mins[:, None]
        maxs = self.channel_maxs[:, None]
        denom = torch.where((maxs - mins) > 0, (maxs - mins), torch.ones_like(maxs))
        x = (x - mins) / denom
        x = x.clamp(0, 1)

        # Build a grayscale image of shape [H=C, W=L] and replicate to RGB.
        gray = (x.numpy() * 255.0).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)  # [C, L, 3]
        return Image.fromarray(rgb)

    def __getitem__(self, idx: int):
        x = self.samples[idx]
        y = self.labels[idx]
        img = self._to_pil(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class LinearClassifier(nn.Module):
    def __init__(self, dim: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


def build_backbone(args) -> Tuple[nn.Module, int]:
    if args.arch not in vits.__dict__.keys():
        raise ValueError("This evaluator currently supports ViT backbones only (vit_tiny/vit_small/vit_base).")

    lora_rank = args.lora_rank if args.use_lora else 0
    model = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0,
        lora_rank=lora_rank,
        lora_alpha=args.lora_alpha,
    )
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))

    if args.checkpoint_path:
        if args.use_lora:
            utils.load_pretrained_weights_for_lora(
                model,
                args.checkpoint_path,
                args.checkpoint_key,
                args.arch,
                args.patch_size,
            )
        else:
            utils.load_pretrained_weights(
                model,
                args.checkpoint_path,
                args.checkpoint_key,
                args.arch,
                args.patch_size,
            )

    return model, embed_dim


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, args, device: torch.device):
    model.eval()
    all_feats = []
    all_labels = []
    for inp, target in loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        intermediate_output = model.get_intermediate_layers(inp, args.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if args.avgpool_patchtokens:
            patch_mean = torch.mean(intermediate_output[-1][:, 1:], dim=1)
            output = torch.cat((output, patch_mean), dim=-1)

        all_feats.append(output)
        all_labels.append(target)

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def _best_label_permutation_accuracy(
    pred_idx: torch.Tensor,
    target: torch.Tensor,
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> Dict[str, object]:
    """Return the best accuracy after remapping predicted labels.

    This solves the optimal assignment between predicted prototype indices and
    ground-truth label ids. It is equivalent to searching over all label
    permutations, but it is exact and scalable.
    """
    pred_idx = pred_idx.detach().cpu().long()
    target = target.detach().cpu().long()
    pred_labels = pred_labels.detach().cpu().long()
    target_labels = target_labels.detach().cpu().long()

    row_index = {int(label.item()): i for i, label in enumerate(pred_labels)}
    col_index = {int(label.item()): i for i, label in enumerate(target_labels)}

    confusion = torch.zeros((len(pred_labels), len(target_labels)), dtype=torch.int64)
    for pred_class, true_label in zip(pred_idx.tolist(), target.tolist()):
        if pred_class in row_index and int(true_label) in col_index:
            confusion[row_index[pred_class], col_index[int(true_label)]] += 1

    cost = (-confusion).numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {int(pred_labels[r].item()): int(target_labels[c].item()) for r, c in zip(row_ind, col_ind)}
    remapped = torch.tensor([mapping.get(int(p), int(p)) for p in pred_idx.tolist()], dtype=target.dtype)
    acc = (remapped == target).float().mean().item() * 100.0

    return {
        "acc1": acc,
        "label_mapping": mapping,
        "predicted_labels": [int(v.item()) for v in pred_labels],
        "target_labels": [int(v.item()) for v in target_labels],
    }


def run_linear_finetune(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    args,
) -> Dict[str, float]:
    num_labels = int(torch.unique(train_labels).numel())
    classifier = LinearClassifier(train_feats.shape[1], num_labels).to(train_feats.device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=args.linear_batch_size, shuffle=True, num_workers=0)

    best_acc = 0.0
    for epoch in range(args.linear_epochs):
        classifier.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            logits = classifier(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_feats)
            acc = _accuracy(test_logits, test_labels)
            best_acc = max(best_acc, acc)

        if args.print_logs and (epoch % max(1, args.linear_epochs // 10) == 0 or epoch == args.linear_epochs - 1):
            avg_loss = loss_sum / max(1, n_batches)
            _log(True, f"[linear] epoch={epoch:03d} loss={avg_loss:.4f} test_acc={acc:.2f}")

    return {"acc1": best_acc}


def run_zero_shot(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    args,
) -> Dict[str, float]:
    # Prototype classifier in normalized feature space.
    train_feats = nn.functional.normalize(train_feats, dim=1)
    test_feats = nn.functional.normalize(test_feats, dim=1)

    unique = torch.unique(train_labels)
    prototypes = []
    for c in unique:
        mask = train_labels == c
        prototypes.append(train_feats[mask].mean(dim=0, keepdim=True))
    prototypes = torch.cat(prototypes, dim=0)
    prototypes = nn.functional.normalize(prototypes, dim=1)

    logits = test_feats @ prototypes.t()
    pred_idx = logits.argmax(dim=1)

    if args.search_label_permutations:
        return _best_label_permutation_accuracy(pred_idx, test_labels, unique, torch.unique(test_labels))

    pred_labels = unique[pred_idx]
    acc = (pred_labels == test_labels).float().mean().item() * 100.0
    return {"acc1": acc, "label_mapping": {int(u.item()): int(u.item()) for u in unique}}


def maybe_run_tsne(
    feats: torch.Tensor,
    labels: torch.Tensor,
    out_path: str,
    enabled: bool,
    print_logs: bool,
    dataset_name: str,
    mode_tag: str,
    checkpoint_tag: str,
) -> None:
    if not enabled:
        return
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except Exception as e:
        _log(print_logs, f"[tsne] skipped: missing dependency ({e})")
        return

    X = feats.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    X = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0)
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8, cmap="tab20", alpha=0.8)
    plt.title(f"t-SNE of test features - dataset {dataset_name} - mode {mode_tag} - ckpt {checkpoint_tag}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    _log(print_logs, f"[tsne] saved: {out_path}")


def evaluate_one_dataset(dataset_dir: str, args) -> Dict[str, float]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    _log(args.print_logs, f"device={device}")

    model, embed_dim = build_backbone(args)
    model = model.to(device)
    model.eval()
    _log(args.print_logs, f"backbone={args.arch}, embed_dim={embed_dim}, use_lora={args.use_lora}")

    eval_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_pt = os.path.join(dataset_dir, "train.pt")
    test_pt = os.path.join(dataset_dir, "test.pt")

    tmp_train = torch.load(train_pt, map_location="cpu")
    mins = tmp_train["samples"].float().amin(dim=(0, 2))
    maxs = tmp_train["samples"].float().amax(dim=(0, 2))

    train_ds = UCRCLSAImageDataset(
        train_pt,
        transform=eval_transform,
        channel_mins=mins,
        channel_maxs=maxs,
    )
    test_ds = UCRCLSAImageDataset(
        test_pt,
        transform=eval_transform,
        channel_mins=mins,
        channel_maxs=maxs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )

    _log(args.print_logs, f"dataset={os.path.basename(dataset_dir)} train={len(train_ds)} test={len(test_ds)}")

    with torch.no_grad():
        train_feats, train_labels = extract_features(model, train_loader, args, device)
        test_feats, test_labels = extract_features(model, test_loader, args, device)

    mode_tag = "linearfinetune" if args.linear_finetune else "zeroshot"
    ds_name = os.path.basename(dataset_dir)
    checkpoint_tag = _sanitize_tag(args.checkpoint_tag) if args.checkpoint_tag else "checkpoint"

    # Generate visualization first when requested, then compute accuracy metrics.
    tsne_out_dir = os.path.join(args.output_dir, "tsne")
    maybe_run_tsne(
        test_feats,
        test_labels,
        os.path.join(tsne_out_dir, f"tsne_{ds_name}_{mode_tag}_{checkpoint_tag}.png"),
        args.compute_tsne,
        args.print_logs,
        ds_name,
        mode_tag,
        checkpoint_tag,
    )

    if args.linear_finetune:
        results = run_linear_finetune(train_feats, train_labels, test_feats, test_labels, args)
        results["mode"] = "linear_finetune"
    else:
        results = run_zero_shot(train_feats, train_labels, test_feats, test_labels, args)
        results["mode"] = "zero_shot_bestperm" if args.search_label_permutations else "zero_shot"

    results["num_classes"] = int(torch.unique(train_labels).numel())
    results["train_size"] = int(train_labels.numel())
    results["test_size"] = int(test_labels.numel())
    return results


def main(args):
    utils.fix_random_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tsne"), exist_ok=True)

    if args.dataset.lower() == "all":
        dataset_names = sorted([
            d for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
            and os.path.isfile(os.path.join(args.data_root, d, "train.pt"))
            and os.path.isfile(os.path.join(args.data_root, d, "test.pt"))
        ])
    else:
        dataset_names = [args.dataset]

    skipped = []
    if not args.include_excluded_datasets:
        kept = []
        for d in dataset_names:
            ds_dir = os.path.join(args.data_root, d)
            train_pt = os.path.join(ds_dir, "train.pt")
            test_pt = os.path.join(ds_dir, "test.pt")
            try:
                train_n, train_l = _dataset_split_meta(train_pt)
                test_n, test_l = _dataset_split_meta(test_pt)
                train_has_nan, train_has_inf = _dataset_has_nan_or_inf(train_pt)
                test_has_nan, test_has_inf = _dataset_has_nan_or_inf(test_pt)
            except Exception as e:
                skipped.append({"dataset": d, "reason": f"meta_check_failed: {e}"})
                continue

            too_small = (train_n < args.min_train_samples_for_eval or test_n < args.min_test_samples_for_eval)
            too_short = (train_l < args.min_sequence_length_for_eval or test_l < args.min_sequence_length_for_eval)
            has_bad_values = (train_has_nan or train_has_inf or test_has_nan or test_has_inf)

            if (args.skip_nan_inf_datasets and has_bad_values) or too_small or too_short:
                reasons = []
                if args.skip_nan_inf_datasets and has_bad_values:
                    reasons.append(
                        "nan_or_inf("
                        f"train_nan={train_has_nan}, train_inf={train_has_inf}, "
                        f"test_nan={test_has_nan}, test_inf={test_has_inf})"
                    )
                if too_small:
                    reasons.append(
                        f"too_small(train={train_n}, test={test_n}, "
                        f"thresholds=train>={args.min_train_samples_for_eval}, test>={args.min_test_samples_for_eval})"
                    )
                if too_short:
                    reasons.append(
                        f"too_short(train_L={train_l}, test_L={test_l}, "
                        f"threshold=L>={args.min_sequence_length_for_eval})"
                    )
                skipped.append({
                    "dataset": d,
                    "reason": " | ".join(reasons),
                })
                continue
            kept.append(d)

        dataset_names = kept
        if skipped:
            _log(args.print_logs, f"skipping {len(skipped)} datasets due to small split sizes")

    all_results = {}
    for name in dataset_names:
        ds_dir = os.path.join(args.data_root, name)
        if not os.path.isdir(ds_dir):
            all_results[name] = {"error": f"missing dataset dir: {ds_dir}"}
            continue
        try:
            _log(args.print_logs, f"\n=== Evaluating {name} ===")
            all_results[name] = evaluate_one_dataset(ds_dir, args)
            _log(args.print_logs, f"acc1={all_results[name]['acc1']:.2f}")
        except Exception as e:
            all_results[name] = {"error": str(e)}

    ok_acc = [v["acc1"] for v in all_results.values() if "acc1" in v]
    summary = {
        "num_datasets": len(dataset_names),
        "num_skipped": len(skipped),
        "num_success": len(ok_acc),
        "mean_acc1": float(np.mean(ok_acc)) if ok_acc else None,
    }

    payload = {
        "config": vars(args),
        "summary": summary,
        "skipped_datasets": skipped,
        "results": all_results,
    }

    out_json = os.path.join(args.output_dir, "json", _result_filename(args))
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    if args.print_logs:
        print(json.dumps(summary, indent=2), flush=True)
        print(f"saved results: {out_json}", flush=True)
    else:
        # "return accuracy results" behavior for scripts/pipes.
        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DinoTraining UCR_CLSA evaluation")

    parser.add_argument("--checkpoint_path", default="", type=str, help="Path to DINO checkpoint (student/teacher state dict).")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key in checkpoint dict ("teacher" or "student").')
    parser.add_argument("--checkpoint_tag", default="", type=str, help="Tag used in output filenames, e.g. pre_lora or checkpoint0003.")

    parser.add_argument("--use_lora", default=False, type=utils.bool_flag, help="Whether the evaluation backbone should include LoRA modules.")
    parser.add_argument("--linear_finetune", default=True, type=utils.bool_flag, help="True: train linear head. False: zero-shot prototype eval.")
    parser.add_argument("--search_label_permutations", default=True, type=utils.bool_flag, help="If true and zero-shot is enabled, search the best label remapping before computing accuracy.")
    parser.add_argument("--print_logs", default=True, type=utils.bool_flag, help="True: verbose logs. False: print JSON result payload only.")
    parser.add_argument("--compute_tsne", default=False, type=utils.bool_flag, help="Whether to compute and save t-SNE for test features.")

    parser.add_argument("--data_root", default="/home/yanivgra/Frequency-masked-Embedding-Inference/datasets_clsa/UCR_CLSA", type=str)
    parser.add_argument("--dataset", default="all", type=str, help='Dataset name or "all".')
    parser.add_argument("--output_dir", default="./eval_ucr_clsa", type=str)
    parser.add_argument("--result_filename", default="", type=str, help="Optional explicit JSON filename. If empty, uses dataset+mode in name.")
    parser.add_argument(
        "--include_excluded_datasets",
        default=False,
        type=utils.bool_flag,
        help="If true, evaluates all datasets regardless of split size. If false, skips too-small datasets.",
    )
    parser.add_argument(
        "--skip_nan_inf_datasets",
        default=True,
        type=utils.bool_flag,
        help="If true, skip datasets containing NaN/Inf values in train/test samples.",
    )
    parser.add_argument("--min_train_samples_for_eval", default=30, type=int, help="Skip datasets with fewer training samples than this threshold.")
    parser.add_argument("--min_test_samples_for_eval", default=30, type=int, help="Skip datasets with fewer test samples than this threshold.")
    parser.add_argument("--min_sequence_length_for_eval", default=30, type=int, help="Skip datasets with sequence length shorter than this threshold.")

    parser.add_argument("--arch", default="vit_base", type=str, choices=["vit_tiny", "vit_small", "vit_base"])
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--n_last_blocks", default=1, type=int)
    parser.add_argument("--avgpool_patchtokens", default=True, type=utils.bool_flag)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--linear_batch_size", default=256, type=int)
    parser.add_argument("--linear_epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="", type=str, help="Optional override, e.g. cuda:0 or cpu.")

    args = parser.parse_args()
    main(args)

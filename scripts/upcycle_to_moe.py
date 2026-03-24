"""Convert a dense model checkpoint to Mixture of Experts (MoE).

Upcycling (Komatsuzaki et al., 2023): duplicate FFN blocks into experts,
add noise, initialize gating randomly. Fine-tune with 10-20% of original
training compute.

Usage:
    .venv/Scripts/python scripts/upcycle_to_moe.py \\
        --checkpoint checkpoints/4080_max/latest \\
        --config configs/4080_max.yaml \\
        --num-experts 8 \\
        --num-shared 1 \\
        --output checkpoints/4080_max_moe/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from safetensors.torch import save_file

from cola_coder.cli import cli
from cola_coder.model.config import Config
from cola_coder.model.transformer import Transformer
from cola_coder.training.checkpoint import load_model_only


def upcycle(
    checkpoint_dir: str,
    config_path: str,
    num_experts: int = 8,
    num_shared_experts: int = 1,
    top_k: int = 2,
    noise_std: float = 0.01,
    output_dir: str = "checkpoints/moe",
) -> None:
    """Convert dense checkpoint to MoE.

    Strategy (Komatsuzaki et al., 2023 "Sparse Upcycling"):
    1. Copy each FFN block's weights into N routed experts, adding small noise
       so experts differentiate during fine-tuning.
    2. Copy the FFN weights (without noise) into the shared expert(s) so they
       start with a good initialization for common patterns.
    3. Initialize the gating network randomly (small std) so routing is
       approximately uniform at the start of fine-tuning.
    4. Save as safetensors with output.weight excluded (weight tying).

    Fine-tune the resulting checkpoint with 10–20% of original training compute.
    """
    cli.header("MoE Upcycling", "Dense -> Sparse Mixture of Experts")

    # Load config
    config = Config.from_yaml(config_path)
    cli.info("Source config", config.model.total_params_human)

    # Load dense model onto CPU to avoid VRAM limits during upcycling
    cli.step(1, 5, "Loading dense model")
    device = "cpu"
    model = Transformer(config.model).to(device)
    load_model_only(checkpoint_dir, model, device=device)
    model.eval()

    # Snapshot the full state dict (weight tying already resolved by the model)
    cli.step(2, 5, "Extracting FFN weights")
    state_dict = model.state_dict()

    # Build MoE state dict by expanding every FFN block
    cli.step(3, 5, f"Duplicating FFN -> {num_experts} routed experts + {num_shared_experts} shared")
    moe_state: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if ".ffn." in key:
            # Split on ".ffn." to get the block prefix and the weight name
            # e.g. "blocks.3.ffn.gate_proj.weight" ->
            #      block_prefix = "blocks.3", ffn_suffix = "gate_proj.weight"
            parts = key.split(".ffn.", 1)
            block_prefix = parts[0]
            ffn_suffix = parts[1]

            # Shared experts: clean copy of original weights (no noise)
            for e in range(num_shared_experts):
                shared_key = f"{block_prefix}.ffn.shared_experts.{e}.{ffn_suffix}"
                moe_state[shared_key] = value.clone()

            # Routed experts: copy + small noise so they diverge during fine-tuning
            for e in range(num_experts):
                expert_key = f"{block_prefix}.ffn.experts.{e}.{ffn_suffix}"
                noise = torch.randn_like(value) * noise_std * value.std().clamp(min=1e-8)
                moe_state[expert_key] = value.clone() + noise

            # Gating network: random init (small std => approx uniform routing at start)
            gate_key = f"{block_prefix}.ffn.router.gate.weight"
            if gate_key not in moe_state:
                gate_weight = torch.randn(num_experts, config.model.dim) * 0.01
                moe_state[gate_key] = gate_weight
        else:
            # Non-FFN weights (attention, norms, embeddings): copy as-is
            moe_state[key] = value.clone()

    # Save checkpoint
    cli.step(4, 5, "Saving MoE checkpoint")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Exclude output.weight — re-tied to tok_emb.weight on model load
    moe_state.pop("output.weight", None)

    # Strip _orig_mod. prefix if the source was a torch.compile'd model
    clean_state: dict[str, torch.Tensor] = {}
    for k, v in moe_state.items():
        clean_key = k.replace("_orig_mod.", "")
        clean_state[clean_key] = v

    save_file(clean_state, str(output_path / "model.safetensors"))

    # Write a sidecar JSON so the MoE config is reproducible later
    moe_config_record = {
        "num_experts": num_experts,
        "num_shared_experts": num_shared_experts,
        "top_k": top_k,
        "noise_std": noise_std,
        "source_checkpoint": str(checkpoint_dir),
        "source_config": str(config_path),
    }
    with open(output_path / "moe_config.json", "w") as f:
        json.dump(moe_config_record, f, indent=2)

    # Final statistics
    cli.step(5, 5, "Computing statistics")
    total_params = sum(v.numel() for v in clean_state.values())
    expert_params = sum(v.numel() for k, v in clean_state.items() if "expert" in k)
    shared_params = sum(v.numel() for k, v in clean_state.items() if "shared_experts" in k)

    cli.done(
        "MoE upcycling complete",
        {
            "Output": str(output_path),
            "Total params": f"{total_params / 1e6:.0f}M",
            "Routed expert params": f"{expert_params / 1e6:.0f}M",
            "Shared expert params": f"{shared_params / 1e6:.0f}M",
            "Experts": f"{num_experts} routed + {num_shared_experts} shared",
            "Top-k": str(top_k),
            "Active params per token": (
                f"~{config.model.total_params / 1e6:.0f}M (same compute as dense)"
            ),
            "Next step": "Fine-tune for 10-20% of original training compute",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upcycle a dense cola-coder checkpoint to a Mixture of Experts model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Dense checkpoint path or 'latest' file")
    parser.add_argument("--config", required=True, help="Model config YAML (e.g. configs/4080_max.yaml)")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of routed experts per layer")
    parser.add_argument(
        "--num-shared", type=int, default=1, help="Always-active shared experts (DeepSeek-MoE style)"
    )
    parser.add_argument("--top-k", type=int, default=2, help="Routed experts activated per token")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Std of noise added to expert copies (fraction of weight std)",
    )
    parser.add_argument("--output", default="checkpoints/moe", help="Output directory for MoE checkpoint")

    args = parser.parse_args()

    upcycle(
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        num_experts=args.num_experts,
        num_shared_experts=args.num_shared,
        top_k=args.top_k,
        noise_std=args.noise_std,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

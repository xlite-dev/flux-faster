import argparse
import os


def create_parser():
    """Creates CLI args parser."""
    parser = argparse.ArgumentParser()

    # general options
    parser.add_argument("--ckpt", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--prompt", type=str, default="A cat playing with a ball of yarn")
    parser.add_argument("--cache-dir", type=str, default=os.path.expandvars("$HOME/.cache/flux-fast"))
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=4)

    # optimizations - all are on by default but each can be disabled
    parser.add_argument("--disable_bf16", action="store_true")
    # torch.compile OR torch.export + AOTI OR neither
    parser.add_argument("--compile_export_mode", type=str, default="export_aoti",
                        choices=["compile", "export_aoti", "disabled"])
    # fused (q, k, v) projections
    parser.add_argument("--disable_fused_projections", action="store_true")
    # channels_last memory format
    parser.add_argument("--disable_channels_last", action="store_true")
    # Flash Attention v3
    parser.add_argument("--disable_fa3", action="store_true")
    # dynamic float8 quantization
    parser.add_argument("--disable_quant", action="store_true")
    # flags for tuning inductor
    parser.add_argument("--disable_inductor_tuning_flags", action="store_true")
    return parser

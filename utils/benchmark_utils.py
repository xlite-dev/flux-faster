import argparse
import functools
import os
from torch.profiler import record_function


def create_parser():
    """Creates CLI args parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # general options
    parser.add_argument(
        "--ckpt",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        choices=[
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-Kontext-dev",
        ],
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat playing with a ball of yarn",
        help="Text prompt",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Image to use for Kontext"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expandvars("$HOME/.cache/flux-fast"),
        help="Cache directory for storing exported models",
    )
    parser.add_argument(
        "--use-cached-model",
        action="store_true",
        help="Attempt to use cached model only (don't re-export)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.png",
        help="Output image file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to use"
    )
    # file path for optional output PyTorch Profiler trace
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Output PyTorch Profiler trace file path",
    )

    # optimizations - all are on by default but each can be disabled
    parser.add_argument(
        "--disable_bf16",
        action="store_true",
        help="Disables usage of torch.bfloat16",
    )
    # torch.compile OR torch.export + AOTI OR neither
    parser.add_argument(
        "--compile_export_mode",
        type=str,
        default="export_aoti",
        choices=["compile", "export_aoti", "disabled"],
        help="Configures how torch.compile or torch.export + AOTI are used",
    )
    # fused (q, k, v) projections
    parser.add_argument(
        "--disable_fused_projections",
        action="store_true",
        help="Disables fused q,k,v projections",
    )
    # channels_last memory format
    parser.add_argument(
        "--disable_channels_last",
        action="store_true",
        help="Disables usage of torch.channels_last memory format",
    )
    # Flash Attention v3
    parser.add_argument(
        "--disable_fa3",
        action="store_true",
        help="Disables use of Flash Attention V3",
    )
    # dynamic float8 quantization
    parser.add_argument(
        "--disable_quant",
        action="store_true",
        help="Disables usage of dynamic float8 quantization",
    )
    # flags for tuning inductor
    parser.add_argument(
        "--disable_inductor_tuning_flags",
        action="store_true",
        help="Disables use of inductor tuning flags",
    )
    # cache-dit: DBCache configs
    parser.add_argument(
        "--cache_dit_config",
        type=str,
        default=None,
        help="Cache options config of cache-dit: DBCache",
    )
    return parser


# helper to annotate a function within a profiler trace
def annotate(f, title):
    @functools.wraps(f)
    def _f(*args, **kwargs):
        with record_function(title):
            return f(*args, **kwargs)

    return _f

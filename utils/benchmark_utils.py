import argparse
import functools
import os
from torch.profiler import record_function


def create_parser():
    """Creates CLI args parser."""
    parser = argparse.ArgumentParser()

    # general options
    parser.add_argument("--ckpt", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--prompt", type=str, default="A cat playing with a ball of yarn")
    parser.add_argument("--cache-dir", type=str, default=os.path.expandvars("$HOME/.cache/flux-fast"))
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--output-file", type=str, default="output.png")
    # file path for optional output PyTorch Profiler trace
    parser.add_argument("--trace-file", type=str, default=None)

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


# helper to annotate a function within a profiler trace
def annotate(f, title):
    @functools.wraps(f)
    def _f(*args, **kwargs):
        with record_function(title):
            return f(*args, **kwargs)
    return _f

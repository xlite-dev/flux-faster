import argparse
from diffusers import FluxPipeline
import torch
import os
from utils.pipeline_utils import load_package


@torch.no_grad()
def load_pipeline(args):
    pipeline = FluxPipeline.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).to("cuda")

    is_timestep_distilled = not pipeline.transformer.config.guidance_embeds

    transformer_package_path = os.path.join(
        args.cache_dir, "exported_transformer.pt2" if is_timestep_distilled else "exported_dev_transformer.pt"
    )
    decoder_package_path = os.path.join(
        args.cache_dir, "exported_decoder.pt2" if is_timestep_distilled else "exported_dev_decoder.pt"
    )
    loaded_transformer = load_package(transformer_package_path)
    loaded_decoder = load_package(decoder_package_path)
    pipeline.transformer.forward = loaded_transformer
    pipeline.vae.decode = loaded_decoder

    return pipeline


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expandvars("$HOME/.cache/flux-fast"),
        help="Directory where we should expect to fine the AOT exported artifacts as well as the model params.",
    )
    parser.add_argument("--ckpt", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--prompt", type=str, default="A cat playing with a ball of yarn")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Ignored when using Schnell.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="output.png", help="Output image file path")
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    pipeline = load_pipeline(args)

    is_timestep_distilled = not pipeline.transformer.config.guidance_embeds
    image = pipeline(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=256 if is_timestep_distilled else 512,
        guidance_scale=None if is_timestep_distilled else args.guidance_scale,
        generator=torch.manual_seed(args.seed),
    ).images[0]
    image.save(args.output_file)
    print(f"Image serialized to {args.output_file}")

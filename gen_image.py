import random
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from utils.benchmark_utils import annotate, create_parser
from utils.pipeline_utils import load_pipeline  # noqa: E402


def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    pipeline = load_pipeline(args)
    set_rand_seeds(args.seed)

    image = pipeline(
        args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=0.0
    ).images[0]
    image.save(args.output_file)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    # use the cached model to minimize latency
    args.use_cached_model = True
    main(args)

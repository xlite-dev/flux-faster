import random
import time
import torch
from utils.benchmark_utils import create_parser
from utils.pipeline_utils import load_pipeline  # noqa: E402


def main(args):
    pipeline = load_pipeline(args)

    # warmup
    for _ in range(3):
        image = pipeline(
            args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=0.0
        ).images[0]

    # run inference 10 times and compute mean / variance
    timings = []
    for _ in range(10):
        begin = time.time()
        image = pipeline(
            args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=0.0
        ).images[0]
        end = time.time()
        timings.append(end - begin)
    timings = torch.tensor(timings)
    print('time mean/var:', timings, timings.mean().item(), timings.var().item())
    image.save("output.png")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    parser = create_parser()
    args = parser.parse_args()
    main(args)

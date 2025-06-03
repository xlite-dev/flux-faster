import random
import time
import torch
from utils.pipeline_utils import load_pipeline  # noqa: E402


# TODO: Update this to match diffusion-fast, make things more configurable via arg parser, etc.
def main():
    pipeline = load_pipeline({})

    # warmup
    prompt = "A cat playing with a ball of yarn"
    for _ in range(3):
        image = pipeline(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    # run inference 10 times and compute mean / variance
    timings = []
    for _ in range(10):
        begin = time.time()
        image = pipeline(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        end = time.time()
        timings.append(end - begin)
    timings = torch.tensor(timings)
    print('time mean/var:', timings, timings.mean().item(), timings.var().item())
    image.save("output.png")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    main()

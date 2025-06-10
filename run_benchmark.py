import random
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from utils.benchmark_utils import annotate, create_parser
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
    image.save(args.output_file)

    # optionally generate PyTorch Profiler trace
    # this is done after benchmarking because tracing introduces overhead
    if args.trace_file is not None:
        # annotate parts of the model within the profiler trace
        pipeline.transformer.forward = annotate(pipeline.transformer.forward, "denoising_step")
        pipeline.vae.decode = annotate(pipeline.vae.decode, "decoding")
        pipeline.encode_prompt = annotate(pipeline.encode_prompt, "prompt_encoding")
        pipeline.image_processor.postprocess = annotate(
            pipeline.image_processor.postprocess, "postprocessing"
        )
        pipeline.image_processor.numpy_to_pil = annotate(
            pipeline.image_processor.numpy_to_pil, "pil_conversion"
        )

        # Generate trace with the PyTorch Profiler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("timed_region"):
                image = pipeline(
                    args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=0.0
                ).images[0]
        prof.export_chrome_trace(args.trace_file)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    parser = create_parser()
    args = parser.parse_args()
    main(args)

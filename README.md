# flux-fast
Making Flux go brrr on GPUs.

## Results (FLUX.1-schnell)
![flux_graph](https://github.com/user-attachments/assets/3f09147e-bf3c-4d48-a0ba-fcff6fc14d45)

Summary of the optimizations:
* Running with the bfloat16 precision
* `torch.compile`
* Combining q,k,v projections for attention computation
* `torch.channels_last` memory format for the transformer
* Flash Attention v3 (FA3) with (unscaled) conversion of inputs to `torch.float8_e4m3fn`
* Dynamic float8 quantization and quantization of Linear layer weights via `torchao`'s `float8_dynamic_activation_float8_weight`
* Inductor flags:
    * `conv_1x1_as_mm = True`
    * `epilogue_fusion = False`
    * `coordinate_descent_tuning = True`
    * `coordinate_descent_check_all_directions = True`
* `torch.export` + Ahead-of-time Inductor (AOTI) + CUDAGraphs

## Setup
We rely on pure PyTorch for the optimizations. Specifically, we rely on a recent nightly version of PyTorch (e.g. `torch==2.8.0.dev20250605+cu126`).

For hardware, we used a 96GB 700W H100 GPU. Some of the optimizations applied (BFloat16, torch.compile, Combining q,k,v projections, dynamic float8 quantization) are available on CPU as well.

## Running a benchmarking experiment
[`run_benchmark.py`](./run_benchmark.py) is the main script for benchmarking the different optimization techniques. After an experiment has been done, you should expect to see two files:

* A `.csv` file with all the benchmarking numbers. TODO: do this instead of printing to STDOUT
* A `.png` image file corresponding to the experiment.

## Improvements, progressively
<details>
  <summary>Baseline</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

With this, we're at:

TODO: Add plot!

</details>

TODO: Add the rest!

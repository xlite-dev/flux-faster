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
We rely on pure PyTorch for the optimizations. Currently, a relatively recent nightly version of PyTorch is required.
The numbers reported here were gathered using:
* `torch==2.8.0.dev20250605+cu126`
* `diffusers==0.33.1`

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

TODO: Add plot!

</details>

<details>
  <summary>BFloat16</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

<details>
  <summary>torch.compile</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Compile the compute-intensive portions of the model: denoising transformer / decoder
# "max-autotune" mode tunes kernel hyperparameters and applies CUDAGraphs
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)
pipeline.vae.decode = torch.compile(
    pipeline.vae.decode, mode="max-autotune", fullgraph=True
)

# warmup for a few iterations; trigger compilation
for _ in range(3):
    pipeline(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=4,
    ).images[0]

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

<details>
  <summary>Combining attention projection matrices</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.transformer = pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae = pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

Note that `torch.compile` is able to perform this fusion automatically, so we do not
observe a speedup from the fusion (outside of noise) when `torch.compile` is enabled.

TODO: Add plot!

</details>

<details>
  <summary>channels_last memory format</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

<details>
  <summary>Flash Attention V3</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

<details>
  <summary>float8 quantization</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# Apply float8 quantization on weights and activations
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

quantize_(
    pipeline.transformer,
    float8_dynamic_activation_float8_weight(),
)

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

<details>
  <summary>Inductor tuning flags</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# Apply float8 quantization on weights and activations
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

quantize_(
    pipeline.transformer,
    float8_dynamic_activation_float8_weight(),
)

# Tune Inductor flags
config = torch._inductor.config
config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
# adjust autotuning algorithm
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

<details>
  <summary>torch.export + Ahead-Of-Time Inductor (AOTI)</summary>

  To avoid initial compilation times, we can use `torch.export` + Ahead-Of-Time Inductor (AOTI). This will
  serialize a binary, precompiled form of the model without initial compilation overhead.

```python
# Apply torch.export + AOTI. If serialize=True, writes out the exported models within the cache_dir.
# Otherwise, attempts to load previously-exported models from the cache_dir.
# This function also applies CUDAGraphs on the loaded models.
def use_export_aoti(pipeline, cache_dir, serialize=False):
    from torch._inductor.package import load_package

    # create cache dir if needed
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _example_tensor(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    # === Transformer export ===
    # torch.export requires a representative set of example args to be passed in
    transformer_kwargs = {
        "hidden_states": _example_tensor(1, 4096, 64),
        "timestep": torch.tensor([1.], device="cuda", dtype=torch.bfloat16),
        "guidance": None,
        "pooled_projections": _example_tensor(1, 768),
        "encoder_hidden_states": _example_tensor(1, 512, 4096),
        "txt_ids": _example_tensor(512, 3),
        "img_ids": _example_tensor(4096, 3),
        "joint_attention_kwargs": {},
        "return_dict": False,
    }

    # Possibly serialize model out
    transformer_package_path = os.path.join(cache_dir, "exported_transformer.pt2")
    if serialize:
        # Apply export
        exported_transformer: torch.export.ExportedProgram = torch.export.export(
            pipeline.transformer, args=(), kwargs=transformer_kwargs
        )

        # Apply AOTI
        path = torch._inductor.aoti_compile_and_package(
            exported_transformer,
            package_path=transformer_package_path,
            inductor_configs={"max_autotune": True, "triton.cudagraphs": True},
        )

    loaded_transformer = load_package(
        transformer_package_path, run_single_threaded=True
    )

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_transformer(**transformer_kwargs)

    # Apply CUDAGraphs. CUDAGraphs are utilized in torch.compile with mode="max-autotune", but
    # they must be manually applied for torch.export + AOTI.
    loaded_transformer = cudagraph(loaded_transformer)
    pipeline.transformer.forward = loaded_transformer

    # warmup after cudagraphing
    with torch.no_grad():
        pipeline.transformer(**transformer_kwargs)

    # hack to get around export's limitations
    pipeline.vae.forward = pipeline.vae.decode

    vae_decode_kwargs = {
        "return_dict": False,
    }

    # Possibly serialize model out
    decoder_package_path = os.path.join(cache_dir, "exported_decoder.pt2")
    if serialize:
        # Apply export
        exported_decoder: torch.export.ExportedProgram = torch.export.export(
            pipeline.vae, args=(_example_tensor(1, 16, 128, 128),), kwargs=vae_decode_kwargs
        )

        # Apply AOTI
        path = torch._inductor.aoti_compile_and_package(
            exported_decoder,
            package_path=decoder_package_path,
            inductor_configs={"max_autotune": True, "triton.cudagraphs": True},
        )

    loaded_decoder = load_package(decoder_package_path, run_single_threaded=True)

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_decoder(_example_tensor(1, 16, 128, 128), **vae_decode_kwargs)

    loaded_decoder = cudagraph(loaded_decoder)
    pipeline.vae.decode = loaded_decoder

    # warmup for a few iterations
    for _ in range(3):
        pipeline(
            "dummy prompt to trigger torch compilation",
            output_type="pil",
            num_inference_steps=4,
        ).images[0]

    return pipeline
```

Note that, unlike for `torch.compile`, running a model loaded from the torch.export + AOTI workflow
doesn't use CUDAGraphs by default. This was found to result in a ~5% performance decrease vs. torch.compile.
To address this discrepancy, we manually record / replay CUDAGraphs over the exported models using the following helper:
```python
# wrapper to automatically handle CUDAGraph record / replay over the given function
def cudagraph(f):
    from torch.utils._pytree import tree_map_only

    _graphs = {}
    def f_(*args, **kwargs):
        key = hash(tuple(tuple(kwargs[a].shape) for a in sorted(kwargs.keys())
                         if isinstance(kwargs[a], torch.Tensor)))
        if key in _graphs:
            # use the cached wrapper if one exists. this will perform CUDAGraph replay
            wrapped, *_ = _graphs[key]
            return wrapped(*args, **kwargs)

        # record a new CUDAGraph and cache it for future use
        g = torch.cuda.CUDAGraph()
        in_args, in_kwargs = tree_map_only(torch.Tensor, lambda t: t.clone(), (args, kwargs))
        f(*in_args, **in_kwargs) # stream warmup
        with torch.cuda.graph(g):
            out_tensors = f(*in_args, **in_kwargs)
        def wrapped(*args, **kwargs):
            # note that CUDAGraphs require inputs / outputs to be in fixed memory locations.
            # inputs must be copied into the fixed input memory locations.
            [a.copy_(b) for a, b in zip(in_args, args) if isinstance(a, torch.Tensor)]
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    in_kwargs[key].copy_(kwargs[key])
            g.replay()
            # clone() outputs on the way out to disconnect them from the fixed output memory
            # locations. this allows for CUDAGraph reuse without accidentally overwriting memory
            return [o.clone() for o in out_tensors]

        # cache function that does CUDAGraph replay
        _graphs[key] = (wrapped, g, in_args, in_kwargs, out_tensors)
        return wrapped(*args, **kwargs)
    return f_
```

Finally, here is the fully-optimized form of the model:

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# Apply float8 quantization on weights and activations
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

quantize_(
    pipeline.transformer,
    float8_dynamic_activation_float8_weight(),
)

# Tune Inductor flags
config = torch._inductor.config
config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
# adjust autotuning algorithm
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

# Apply torch.export + AOTI with CUDAGraphs
pipeline = use_export_aoti(pipeline, cache_dir=args.cache_dir, serialize=False)

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

TODO: Add plot!

</details>

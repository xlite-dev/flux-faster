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

TBD: Installation / benchmarking instructions, detailed discussion on incremental optimizations, etc.

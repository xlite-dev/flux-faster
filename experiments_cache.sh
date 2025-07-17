#!/bin/bash

# TODO: FA3, H100
CKPT="black-forest-labs/FLUX.1-dev"

# bfloat16
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16.json.gz \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --output-file bf16.png \
    > bf16.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file optimized.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --output-file optimized.png \
    > optimized.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags 
# + cache_dit
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file optimized_cache_dit.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --cache_dit_config cache_config.yaml \
    --output-file optimized_cache_dit.png \
    > optimized_cache_dit.txt 2>&1

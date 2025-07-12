#!/bin/bash

CKPT="black-forest-labs/FLUX.1-dev"

# baseline
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file baseline.json.gz \
    --disable_bf16 \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --output-file baseline.png \
    > baseline.txt 2>&1

# bfloat16
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bfloat16.json.gz \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --output-file bf16.png \
    > bf16.txt 2>&1

# bfloat16 + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache.json.gz \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache.png \
    > bf16_cache.txt 2>&1

# bfloat16 + torch.compile
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile.json.gz \
    --compile_export_mode compile \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --output-file bf16_compile.png \
    > bf16_compile.txt 2>&1

# bfloat16 + torch.compile + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile.json.gz \
    --compile_export_mode compile \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile.png \
    > bf16_cache_compile.txt 2>&1

# bfloat16 + torch.compile + qkv projection + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv.json.gz \
    --compile_export_mode compile \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv.png \
    > bf16_cache_compile_qkv.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan.png \
    > bf16_cache_compile_qkv_chan.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_fa3.json.gz \
    --compile_export_mode compile \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_fa3.png \
    > bf16_cache_compile_qkv_chan_fa3.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + float8 quant + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_fa3_quant.json.gz \
    --compile_export_mode compile \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_fa3_quant.png \
    > bf16_cache_compile_qkv_chan_fa3_quant.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + float8 quant + inductor flags + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_fa3_quant_flags.json.gz \
    --compile_export_mode compile \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_fa3_quant_flags.png \
    > bf16_cache_compile_qkv_chan_fa3_quant_flags.txt 2>&1

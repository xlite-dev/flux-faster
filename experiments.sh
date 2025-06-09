#!/bin/bash

# baseline
python run_benchmark.py \
    --trace-file baseline.json.gz \
    --disable_bf16 \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags

# bfloat16
python run_benchmark.py \
    --trace-file bfloat16.json.gz \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags

# bfloat16 + torch.compile
python run_benchmark.py \
    --trace-file bf16_compile.json.gz \
    --compile_export_mode compile \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags

# bfloat16 + torch.compile + qkv projection
python run_benchmark.py \
    --trace-file bf16_compile_qkv.json.gz \
    --compile_export_mode compile \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags

# bfloat16 + torch.compile + qkv projection + channels_last
python run_benchmark.py \
    --trace-file bf16_compile_qkv_chan.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags

# bfloat16 + torch.compile + qkv projection + channels_last + FA3
python run_benchmark.py \
    --trace-file bf16_compile_qkv_chan_fa3.json.gz \
    --compile_export_mode compile \
    --disable_quant \
    --disable_inductor_tuning_flags

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + float8 quant
python run_benchmark.py \
    --trace-file bf16_compile_qkv_chan_fa3_quant.json.gz \
    --compile_export_mode compile \
    --disable_inductor_tuning_flags

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + float8 quant + inductor flags
python run_benchmark.py \
    --trace-file bf16_compile_qkv_chan_fa3_quant_flags.json.gz \
    --compile_export_mode compile

# fully optimized (torch.export + AOTI to address cold start)
python run_benchmark.py --trace-file fully_optimized.json.gz

#!/bin/bash

# TODO: FA3, H100
CKPT="black-forest-labs/FLUX.1-dev"

# ============== ROW 1 ==============
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

# bfloat16 + cache: F12B12 + warmup 8 steps + max cached 8 steps
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F12B12W8M8.json.gz \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 12 --Bn 12 \
    --warmup_steps 8 \
    --max_cached_steps 8 \
    --output-file bf16_cache_F12B12W8M8.png \
    > bf16_cache_F12B12W8M8.txt 2>&1

# bfloat16 + torch.compile + cache: F12B12 + warmup 8 steps + max cached 8 steps
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F12B12W8M8_compile.json.gz \
    --compile_export_mode compile \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 12 --Bn 12 \
    --warmup_steps 8 \
    --max_cached_steps 8 \
    --output-file bf16_cache_F12B12W8M8_compile.png \
    > bf16_cache_F12B12W8M8_compile.txt 2>&1


# ============== ROW 2 ==============
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

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --output-file bf16_compile_qkv_chan_quant_flags.png \
    > bf16_compile_qkv_chan_quant_flags.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags 
# + cache: F12B12 + warmup 8 steps + max cached 8 steps
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F12B12W8M8_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_F12B12W8M8_compile_qkv_chan_quant_flags.png \
    > bf16_cache_F12B12W8M8_compile_qkv_chan_quant_flags.txt 2>&1


# ============== ROW 3 ==============
# bfloat16 + only compile transformer blocks
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --output-file bf16_compile_trn.png \
    > bf16_compile_trn.txt 2>&1

# bfloat16 + only compile transformer blocks + qkv projection + channels_last + float8 quant + inductor flags
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --output-file bf16_compile_qkv_chan_quant_flags_trn.png \
    > bf16_compile_qkv_chan_quant_flags_trn.txt 2>&1

# bfloat16 + only compile transformer blocks + qkv projection + channels_last + float8 quant + inductor flags 
# + cache: F12B12 + warmup 8 steps + max cached 8 steps
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F12B12W8M8_compile_qkv_chan_quant_flags_trn.txt 2>&1


# ============== ROW 4 ==============
# bfloat16 + only compile transformer blocks + qkv projection + channels_last + float8 quant + inductor flags
# + cache: F8B0 + no warmup steps + no limit cached steps
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F8B0W0M0_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 8 --Bn 0 \
    --warmup_steps 0 \
    --max_cached_steps -1 \
    --output-file bf16_cache_F8B0W0M0_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F8B0W0M0_compile_qkv_chan_quant_flags_trn.txt 2>&1
    
# bfloat16 + only compile transformer blocks + qkv projection + channels_last + float8 quant + inductor flags 
# + cache: F1B0 + no warmup steps + no limit cached steps
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F1B0W0M0_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 1 --Bn 0 \
    --warmup_steps 0 \
    --max_cached_steps -1 \
    --output-file bf16_cache_F1B0W0M0_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F1B0W0M0_compile_qkv_chan_quant_flags_trn.txt 2>&1

# bfloat16 + only compile transformer blocks + qkv projection + channels_last + float8 quant + inductor flags 
# + cache: F1B0 + no warmup steps + no limit cached steps + TaylorSeer
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F1B0W0M0_taylorseer_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 1 --Bn 0 \
    --warmup_steps 0 \
    --max_cached_steps -1 \
    --enable_taylorsser \
    --output-file bf16_cache_F1B0W0M0_taylorseer_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F1B0W0M0_taylorseer_compile_qkv_chan_quant_flags_trn.txt 2>&1

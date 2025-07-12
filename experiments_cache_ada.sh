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
# L20: 20.24 PSNR: 19.28
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

# only compile transformer blocks
# L20: 20.49 PSNR: 39.72
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --output-file bf16_compile_trn.png \
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
    --trace-file bf16_compile_qkv.json.gz \
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

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + cache
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_quant.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --disable_inductor_tuning_flags \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_quant.png \
    > bf16_cache_compile_qkv_chan_quant.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags
# L20 13.29 PSNR: 18.07
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --output-file bf16_compile_qkv_chan_quant_flags.png \
    > bf16_compile_qkv_chan_quant_flags.txt 2>&1

# only compile transformer blocks
# L20 13.26 PSNR: 21.77
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --output-file bf16_compile_qkv_chan_quant_flags_trn.png \
    > bf16_compile_qkv_chan_quant_flags.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags + cache
# L20: 11.21 PSNR: 22.24
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_quant_flags.png \
    > bf16_cache_compile_qkv_chan_quant_flags.txt 2>&1

# only compile transformer blocks
# L20: 11.14 PSNR: 21.89
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_compile_qkv_chan_quant_flags.txt 2>&1

# only compile transformer blocks + F8B0
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F8B0_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 8 --Bn 0 \
    --output-file bf16_cache_F8B0_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F8B0_compile_qkv_chan_quant_flags_trn.txt 2>&1

# only compile transformer blocks + F8B0 + no warmup + no limit cache steps
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
    
# only compile transformer blocks + F1B0
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F1B0_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 1 --Bn 0 \
    --output-file bf16_cache_F1B0_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F1B0_compile_qkv_chan_quant_flags_trn.txt 2>&1

# only compile transformer blocks + F1B0 + taylorseer
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_F1B0_taylorseer_compile_qkv_chan_quant_flags_trn.json.gz \
    --compile_export_mode compile \
    --only_compile_transformer_blocks \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --Fn 1 --Bn 0 \
    --enable_taylorsser \
    --output-file bf16_cache_F1B0_taylorseer_compile_qkv_chan_quant_flags_trn.png \
    > bf16_cache_F1B0_taylorseer_compile_qkv_chan_quant_flags_trn.txt 2>&1

# only compile transformer blocks + F1B0 + no warmup + no limit cache steps
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

# only compile transformer blocks + F1B0 + taylorseer + no warmup + no limit cache steps
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
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
# L20: 13.292539596557617s
# AUTOTUNE scaled_mm(4096x3072, 3072x64, 1, , 64)
#   triton_scaled_mm_bias_29989 0.0317 ms 100.0% ACC_TYPE='tl.float32', BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   _scaled_mm 0.0328 ms 96.9%
#   triton_scaled_mm_bias_29990 0.0328 ms 96.9% ACC_TYPE='tl.float32', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   triton_scaled_mm_bias_29994 0.0348 ms 91.2% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=5, num_warps=2
#   triton_scaled_mm_bias_30040 0.0348 ms 91.2% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=5, num_warps=2
#   triton_scaled_mm_bias_30054 0.0348 ms 91.2% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=6, num_warps=4
#   triton_scaled_mm_bias_29993 0.0358 ms 88.6% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=128, BLOCK_N=32, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   triton_scaled_mm_bias_30042 0.0358 ms 88.6% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=5, num_warps=4
#   triton_scaled_mm_bias_30052 0.0358 ms 88.6% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=6, num_warps=2
#   triton_scaled_mm_bias_30053 0.0369 ms 86.1% ACC_TYPE='tl.float32', BLOCK_K=64, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=6, num_warps=2
# SingleProcess AUTOTUNE benchmarking takes 3.3043 seconds and 0.0007 seconds precompiling for 76 choices
# SingleProcess AUTOTUNE benchmarking takes 0.4329 seconds and 0.0002 seconds precompiling for 7 choices
# time mean/var: tensor([13.3014, 13.2864, 13.2873, 13.2740, 13.2803, 13.2851, 13.2980, 13.2955,
#         13.3052, 13.3123]) 13.292539596557617 0.00014241029566619545
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --output-file bf16_compile_qkv_chan_quant_flags.png \
    > bf16_compile_qkv_chan_quant_flags.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + float8 quant + inductor flags + cache
# L20: 11.214410781860352s
# AUTOTUNE scaled_mm(512x4096, 4096x3072, 1, , 3072)
#   _scaled_mm 0.0768 ms 100.0%
#   triton_scaled_mm_bias_390 0.2591 ms 29.6% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   triton_scaled_mm_bias_391 0.2703 ms 28.4% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   triton_scaled_mm_bias_392 0.2724 ms 28.2% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=128, BLOCK_N=32, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   triton_scaled_mm_bias_393 0.2970 ms 25.9% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=5, num_warps=2
#   triton_scaled_mm_bias_412 0.2980 ms 25.8% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=32, BLOCK_N=128, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=2, num_warps=4
#   triton_scaled_mm_bias_428 0.2980 ms 25.8% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=32, BLOCK_N=128, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=3, num_warps=4
#   triton_scaled_mm_bias_444 0.2980 ms 25.8% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=32, BLOCK_N=128, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=4, num_warps=4
#   triton_scaled_mm_bias_460 0.2980 ms 25.8% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=32, BLOCK_N=128, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=5, num_warps=4
#   triton_scaled_mm_bias_476 0.2980 ms 25.8% ACC_TYPE='tl.float32', BLOCK_K=32, BLOCK_M=32, BLOCK_N=128, EVEN_K=False, GROUP_M=8, SCALING_ROWWISE=False, USE_FAST_ACCUM=True, num_stages=6, num_warps=4
# SingleProcess AUTOTUNE benchmarking takes 6.0628 seconds and 3.3437 seconds precompiling for 98 choices
# time mean/var: tensor([11.1875, 11.1962, 11.2009, 11.2101, 11.2106, 11.2205, 11.2240, 11.2271,
#        11.2318, 11.2353]) 11.214410781860352 0.0002548588381614536
python run_benchmark.py \
    --ckpt ${CKPT} \
    --trace-file bf16_cache_compile_qkv_chan_quant_flags.json.gz \
    --compile_export_mode compile \
    --disable_fa3 \
    --num_inference_steps 28 \
    --enable_cache_dit \
    --output-file bf16_cache_compile_qkv_chan_quant_flags.png \
    > bf16_cache_compile_qkv_chan_quant_flags.txt 2>&1


#!/bin/bash
# gfx1103 (Radeon 780M) GPU 运行脚本

# 清理旧缓存（首次运行或出问题时使用）
if [[ "$1" == "--clean" ]]; then
    echo "清理 MIGraphX 缓存..."
    rm -f /tmp/migraphx_cache.mxr
    shift
fi

# 关键：使用 gfx1100 作为兼容版本
# gfx1100 = RDNA3 桌面版，比 gfx1103 支持更好
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# MIOpen 设置
export MIOPEN_FIND_MODE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=0

# HIP 设置
export HIP_VISIBLE_DEVICES=0
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

# 内存设置（iGPU 共享内存）
export GPU_MAX_HEAP_SIZE=80
export GPU_MAX_ALLOC_PERCENT=80

echo "=========================================="
echo "AMD GPU (gfx1103 -> gfx1100 兼容模式)"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "=========================================="
echo ""
echo "首次运行需要编译模型，请耐心等待 2-5 分钟..."
echo "编译完成后，后续启动会很快（使用缓存）"
echo ""

cd /home/neomelt/sentry_aim_26/build
./tests/armor_detector_test "$@"

#!/bin/bash
# 启动脚本：设置 ONNX Runtime GPU 加速所需的环境变量

export LD_LIBRARY_PATH="/home/neomelt/.local/lib/python3.12/site-packages/onnxruntime/capi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=11.0.3

echo "环境变量已设置："
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# 运行传入的命令
exec "$@"

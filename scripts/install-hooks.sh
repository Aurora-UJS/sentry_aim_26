#!/bin/bash

# Git Pre-commit Hook 安装脚本 (强制 Clang-19 版)
# 适用项目: sentry_aim_26

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# 自动识别项目根目录
if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

# 创建 pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash

# AURORA_AIM Pre-commit Hook
# 强制使用 clang-format-19 检查暂存区代码

CLANG_FORMAT_BIN="clang-format-19"

# 检查工具是否存在
if ! command -v $CLANG_FORMAT_BIN &> /dev/null; then
    echo "❌ 错误: 未找到 $CLANG_FORMAT_BIN"
    echo "请先安装: sudo apt install clang-format-19"
    exit 1
fi

echo "🔍 正在使用 $($CLANG_FORMAT_BIN --version | cut -d' ' -f3) 检查暂存区代码格式..."

# 获取暂存的 C++ 文件 (排除 3rdparty 和 build 路径下的文件)
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|hpp|h|cc)$' | grep -vE '^(3rdparty/|build/|vcpkg_installed/)')

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

FAILED=0
for file in $STAGED_FILES; do
    if [ -f "$file" ]; then
        # 执行 dry-run 检查
        if ! $CLANG_FORMAT_BIN --dry-run --Werror "$file" 2>/dev/null; then
            echo "❌ 格式不合规: $file"
            FAILED=1
        fi
    fi
done

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "========================================================"
    echo "⚠️  代码格式检查失败！"
    echo "检测到您的修改不符合 Clang-format-19 标准。"
    echo "请运行以下命令修复并重新 add："
    echo "  ./scripts/lint.sh format"
    echo "========================================================"
    exit 1
fi

echo "✅ 格式检查通过，准予提交。"
exit 0
EOF

chmod +x "$HOOKS_DIR/pre-commit"
echo "✅ Clang-19 Pre-commit hook 已成功安装至 $HOOKS_DIR"
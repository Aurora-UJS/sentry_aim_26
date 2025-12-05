# Git Pre-commit Hook 安装脚本
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

# 创建 pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash

# AURORA_AIM Pre-commit Hook
# 在提交前检查代码格式

echo "🔍 检查代码格式..."

# 获取暂存的 C++ 文件
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|hpp|h|cc)$')

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

FAILED=0
for file in $STAGED_FILES; do
    if [ -f "$file" ]; then
        if ! clang-format --dry-run --Werror "$file" 2>/dev/null; then
            echo "❌ 格式错误: $file"
            FAILED=1
        fi
    fi
done

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "========================================"
    echo "代码格式检查失败！"
    echo "请运行以下命令修复："
    echo "  ./scripts/lint.sh format"
    echo "然后重新暂存文件"
    echo "========================================"
    exit 1
fi

echo "✅ 代码格式检查通过"
exit 0
EOF

chmod +x "$HOOKS_DIR/pre-commit"
echo "✅ Pre-commit hook 已安装"

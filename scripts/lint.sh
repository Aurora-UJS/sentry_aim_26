#!/bin/bash

# AURORA_AIM 代码格式化和检查脚本
# 用于本地开发和 CI/CD

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 获取项目根目录 (假设脚本在项目根目录或 scripts 子目录下)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

cd "$PROJECT_ROOT"

CLANG_FORMAT_BIN="clang-format-19"
CLANG_TIDY_BIN="clang-tidy-19"

print_info "使用工具: $($CLANG_FORMAT_BIN --version)"

# 查找所有 C++ 源文件 (使用函数返回 find 参数)
find_sources() {
    find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" \) \
        -not -path "./build/*" \
        -not -path "./vcpkg_installed/*" \
        -not -path "./.git/*" \
        -not -path "./vcpkg/*" \
        -not -path "./3rdparty/*" \
        -print0
}

# 格式化代码
format_code() {
    print_info "开始检查并格式化代码..."

    local count=0
    local total=0
    # 临时关闭 set -e 以便统计
    set +e
    while IFS= read -r -d '' file; do
        ((total++))
        # 使用 -i 直接修改，如果文件没变，它的 mtime 不一定会大幅跳动
        # 我们通过 clang-format 的输出判断是否发生了改变（更稳健）
        BEFORE_HASH=$(md5sum "$file" | cut -d' ' -f1)
        $CLANG_FORMAT_BIN -i "$file"
        AFTER_HASH=$(md5sum "$file" | cut -d' ' -f1)

        if [ "$BEFORE_HASH" != "$AFTER_HASH" ]; then
            echo "  [FIXED] $file"
            ((count++))
        fi
    done < <(find_sources)
    set -e

    if [ $count -eq 0 ]; then
        print_info "✅ 所有 $total 个文件均已符合标准。"
    else
        print_info "✅ 格式化完成，修复了 $count 个文件。"
    fi
}

# 检查代码格式
check_format() {
    print_info "正在检查代码格式..."
    
    FAILED=0
    while IFS= read -r -d '' file; do
        if ! $CLANG_FORMAT_BIN --dry-run --Werror "$file" 2>/dev/null; then
            print_error "格式错误: $file"
            FAILED=1
        fi
    done < <(find_sources)
    
    if [ $FAILED -eq 1 ]; then
        print_error "❌ 代码格式检查失败！请运行 './$(basename "$0") format' 修复后提交"
        exit 1
    fi
    
    print_info "✅ 所有文件格式正确"
}

# 运行 clang-tidy
run_tidy() {
    print_info "运行静态分析 (clang-tidy)..."
    
    if [ ! -f "build/compile_commands.json" ]; then
        print_error "未找到 build/compile_commands.json"
        print_warning "提示: 请开启 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 编译项目"
        return 1
    fi
    
    # 只针对 .cpp 文件运行 tidy，.hpp 会被包含在内
    FILES=$(find . -type f -name "*.cpp" -not -path "./build/*" -not -path "./3rdparty/*")
    
    if [ -z "$FILES" ]; then
        print_warning "未找到 .cpp 源文件"
        return 0
    fi
    
    $CLANG_TIDY_BIN -p build $FILES --quiet
    
    print_info "✅ 静态分析完成"
}

# 显示帮助
show_help() {
    echo "用法: $0 <命令>"
    echo ""
    echo "命令:"
    echo "  format    自动格式化所有源代码 (匹配 CI v19 标准)"
    echo "  check     检查代码格式 (仅检查，不修改文件)"
    echo "  tidy      运行 clang-tidy 静态分析"
    echo "  all       运行格式检查 + 静态分析"
    echo "  help      显示此帮助信息"
}

# 主函数
case "${1:-help}" in
    format)
        format_code
        ;;
    check)
        check_format
        ;;
    tidy)
        run_tidy
        ;;
    all)
        check_format && run_tidy
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "未知命令: $1"
        show_help
        exit 1
        ;;
esac
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

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 查找所有 C++ 源文件
find_sources() {
    find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" \) \
        -not -path "./build/*" \
        -not -path "./vcpkg_installed/*" \
        -not -path "./.git/*" \
        -not -path "./vcpkg/*"
}

# 格式化代码
format_code() {
    print_info "格式化代码..."
    
    FILES=$(find_sources)
    
    if [ -z "$FILES" ]; then
        print_warning "未找到 C++ 源文件"
        return 0
    fi
    
    for file in $FILES; do
        clang-format -i "$file"
        echo "  格式化: $file"
    done
    
    print_info "✅ 格式化完成"
}

# 检查代码格式
check_format() {
    print_info "检查代码格式..."
    
    FILES=$(find_sources)
    
    if [ -z "$FILES" ]; then
        print_warning "未找到 C++ 源文件"
        return 0
    fi
    
    FAILED=0
    for file in $FILES; do
        if ! clang-format --dry-run --Werror "$file" 2>/dev/null; then
            print_error "格式错误: $file"
            FAILED=1
        fi
    done
    
    if [ $FAILED -eq 1 ]; then
        print_error "代码格式检查失败！运行 '$0 format' 修复"
        return 1
    fi
    
    print_info "✅ 所有文件格式正确"
}

# 运行 clang-tidy
run_tidy() {
    print_info "运行静态分析..."
    
    if [ ! -f "build/compile_commands.json" ]; then
        print_error "未找到 compile_commands.json，请先构建项目"
        return 1
    fi
    
    FILES=$(find_sources | grep -E "\.cpp$")
    
    if [ -z "$FILES" ]; then
        print_warning "未找到 C++ 源文件"
        return 0
    fi
    
    clang-tidy -p build $FILES 2>&1 | tee clang-tidy-output.txt
    
    if grep -q "error:" clang-tidy-output.txt; then
        print_error "发现代码问题"
        return 1
    fi
    
    print_info "✅ 静态分析通过"
}

# 显示帮助
show_help() {
    echo "用法: $0 <命令>"
    echo ""
    echo "命令:"
    echo "  format    格式化所有源代码"
    echo "  check     检查代码格式（不修改文件）"
    echo "  tidy      运行 clang-tidy 静态分析"
    echo "  all       运行所有检查"
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

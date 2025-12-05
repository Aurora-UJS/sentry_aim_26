#!/bin/bash

# AURORA_AIM 项目环境一键配置脚本
# 工具链要求：clang19+、cmake 3.28+、ninja 1.13.1+、vcpkg、git
# C++ 标准：C++20

set -e  # 遇到错误立即退出

# 获取脚本所在目录和项目根目录（在脚本开始时确定）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测操作系统
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        print_error "无法检测操作系统"
        exit 1
    fi
    print_info "检测到操作系统: $OS $VER"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查版本是否满足要求
version_ge() {
    # 比较版本号 $1 >= $2
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

# 安装基础工具
install_basic_tools() {
    print_info "安装基础工具..."
    
    case $OS in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y wget curl build-essential git pkg-config \
                autoconf automake libtool unzip tar zip
            ;;
        arch|manjaro)
            sudo pacman -Sy --noconfirm wget curl base-devel git pkg-config \
                autoconf automake libtool unzip tar zip
            ;;
        fedora|rhel|centos)
            sudo dnf install -y wget curl gcc gcc-c++ make git pkg-config \
                autoconf automake libtool unzip tar zip
            ;;
        *)
            print_error "不支持的操作系统: $OS"
            exit 1
            ;;
    esac
}

# 安装 Clang 19+
install_clang() {
    print_info "检查 Clang 版本..."
    
    if command_exists clang; then
        CLANG_VERSION=$(clang --version | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1 | cut -d. -f1)
        if [ "$CLANG_VERSION" -ge 19 ]; then
            print_info "Clang $CLANG_VERSION 已安装，满足要求"
            return
        else
            print_warning "Clang $CLANG_VERSION 版本过低，需要 19+"
        fi
    fi
    
    print_info "安装 Clang 19+..."
    
    case $OS in
        ubuntu|debian)
            # 添加 LLVM 官方源
            wget https://apt.llvm.org/llvm.sh
            chmod +x llvm.sh
            sudo ./llvm.sh 19
            rm llvm.sh
            
            # 设置为默认编译器
            sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100
            sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100
            
            # 安装 clang-format 和 clang-tidy
            sudo apt-get install -y clang-format-19 clang-tidy-19
            sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100
            sudo update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-19 100
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm clang
            ;;
        fedora|rhel|centos)
            sudo dnf install -y clang clang-tools-extra
            ;;
    esac
    
    print_info "Clang 安装完成: $(clang --version | head -n1)"
}

# 安装 CMake 3.28+
install_cmake() {
    print_info "检查 CMake 版本..."
    
    if command_exists cmake; then
        CMAKE_VERSION=$(cmake --version | head -n1 | grep -oP '\d+\.\d+\.\d+')
        if version_ge "$CMAKE_VERSION" "3.28.0"; then
            print_info "CMake $CMAKE_VERSION 已安装，满足要求"
            return
        else
            print_warning "CMake $CMAKE_VERSION 版本过低，需要 3.28+"
        fi
    fi
    
    print_info "安装 CMake 3.28+..."
    
    # 从官方下载最新版本
    CMAKE_VERSION="3.28.3"
    CMAKE_DIR="cmake-${CMAKE_VERSION}-linux-x86_64"
    
    cd /tmp
    wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_DIR}.tar.gz"
    tar -xzf "${CMAKE_DIR}.tar.gz"
    
    # 只复制 bin、share 目录，避免 man 目录冲突
    sudo cp -r "${CMAKE_DIR}"/bin/* /usr/local/bin/
    sudo cp -r "${CMAKE_DIR}"/share/* /usr/local/share/
    rm -rf "${CMAKE_DIR}" "${CMAKE_DIR}.tar.gz"
    
    print_info "CMake 安装完成: $(cmake --version | head -n1)"
}

# 安装 Ninja 1.13.1+
install_ninja() {
    print_info "检查 Ninja 版本..."
    
    if command_exists ninja; then
        NINJA_VERSION=$(ninja --version)
        if version_ge "$NINJA_VERSION" "1.13.1"; then
            print_info "Ninja $NINJA_VERSION 已安装，满足要求"
            return
        else
            print_warning "Ninja $NINJA_VERSION 版本过低，需要 1.13.1+"
        fi
    fi
    
    print_info "安装 Ninja 1.13.1+..."
    
    cd /tmp
    wget "https://github.com/ninja-build/ninja/releases/download/v1.13.1/ninja-linux.zip"
    unzip -o ninja-linux.zip
    sudo mv ninja /usr/local/bin/
    sudo chmod +x /usr/local/bin/ninja
    rm ninja-linux.zip
    
    print_info "Ninja 安装完成: $(ninja --version)"
}

# 安装 vcpkg
install_vcpkg() {
    print_info "检查 vcpkg..."
    
    VCPKG_ROOT="${HOME}/vcpkg"
    
    if [ -d "$VCPKG_ROOT" ] && [ -f "$VCPKG_ROOT/vcpkg" ]; then
        print_info "vcpkg 已安装在 $VCPKG_ROOT"
    else
        print_info "安装 vcpkg..."
        
        if [ -d "$VCPKG_ROOT" ]; then
            rm -rf "$VCPKG_ROOT"
        fi
        
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
        cd "$VCPKG_ROOT"
        ./bootstrap-vcpkg.sh
    fi
    
    # 设置环境变量
    if ! grep -q "VCPKG_ROOT" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# vcpkg 环境变量" >> ~/.bashrc
        echo "export VCPKG_ROOT=\"$VCPKG_ROOT\"" >> ~/.bashrc
        echo "export PATH=\"\$VCPKG_ROOT:\$PATH\"" >> ~/.bashrc
    fi
    
    export VCPKG_ROOT="$VCPKG_ROOT"
    export PATH="$VCPKG_ROOT:$PATH"
    
    print_info "vcpkg 安装完成"
}

# 配置编译器环境变量
setup_compiler_env() {
    print_info "配置编译器环境变量..."
    
    ENV_VARS="
# AURORA_AIM 项目编译环境
export CC=clang
export CXX=clang++
export CMAKE_C_COMPILER=clang
export CMAKE_CXX_COMPILER=clang++
export CMAKE_EXPORT_COMPILE_COMMANDS=ON
"
    
    if ! grep -q "AURORA_AIM 项目编译环境" ~/.bashrc; then
        echo "$ENV_VARS" >> ~/.bashrc
    fi
    
    export CC=clang
    export CXX=clang++
    export CMAKE_C_COMPILER=clang
    export CMAKE_CXX_COMPILER=clang++
    
    print_info "编译器环境变量已配置"
}

# 安装项目依赖
install_project_dependencies() {
    print_info "安装项目依赖..."
    
    if [ ! -f "$PROJECT_ROOT/vcpkg.json" ]; then
        print_warning "未找到 vcpkg.json，跳过依赖安装"
        return
    fi
    
    # 常见依赖预安装（系统包）
    print_info "预安装常用系统依赖..."
    case $OS in
        ubuntu|debian)
            sudo apt-get install -y \
                libopencv-dev \
                libeigen3-dev \
                libboost-all-dev \
                libspdlog-dev \
                libyaml-cpp-dev \
                catch2
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm \
                opencv \
                eigen \
                boost \
                spdlog \
                yaml-cpp \
                catch2
            ;;
        fedora|rhel|centos)
            sudo dnf install -y \
                opencv-devel \
                eigen3-devel \
                boost-devel \
                spdlog-devel \
                yaml-cpp-devel \
                catch-devel
            ;;
    esac
    
    # 通过 vcpkg 安装项目依赖
    print_info "通过 vcpkg 安装项目依赖..."
    cd "$PROJECT_ROOT"
    
    if [ -f "vcpkg.json" ]; then
        "$VCPKG_ROOT/vcpkg" install --triplet=x64-linux
    fi
}

# 初始化项目
initialize_project() {
    print_info "初始化项目..."
    
    cd "$PROJECT_ROOT"
    
    # 创建构建目录
    if [ ! -d "build" ]; then
        mkdir -p build
        print_info "创建 build 目录"
    fi
    
    # 配置 CMake（使用 vcpkg toolchain）
    print_info "配置 CMake 项目..."
    cd build
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        ..
    
    # 链接 compile_commands.json 到项目根目录（用于 LSP）
    if [ -f "compile_commands.json" ]; then
        ln -sf "$PROJECT_ROOT/build/compile_commands.json" "$PROJECT_ROOT/compile_commands.json"
        print_info "已链接 compile_commands.json"
    fi
    
    print_info "项目初始化完成"
}

# 主函数
main() {
    print_info "开始配置 AURORA_AIM 项目环境..."
    print_info "======================================"
    
    detect_os
    install_basic_tools
    install_clang
    install_cmake
    install_ninja
    install_vcpkg
    setup_compiler_env
    install_project_dependencies
    initialize_project
    
    print_info "======================================"
    print_info "环境配置完成！"
    print_info ""
    print_info "请运行以下命令使环境变量生效："
    print_info "  source ~/.bashrc"
    print_info ""
    print_info "然后可以开始构建项目："
    print_info "  cd build"
    print_info "  ninja"
    print_info ""
    print_info "工具链版本："
    print_info "  Clang: $(clang --version | head -n1)"
    print_info "  CMake: $(cmake --version | head -n1)"
    print_info "  Ninja: $(ninja --version)"
    print_info "  Git: $(git --version)"
}

# 执行主函数
main

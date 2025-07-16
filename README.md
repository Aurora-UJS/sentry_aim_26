# sentry_aim_26

基于 C++23 的模块化自动瞄准系统空项目。

## 结构说明

``` bash
sentry_aim_26/
.
├── assets
│   └── model
├── CMakeLists.txt
├── config
├── include
│   └── sentry_aim_26
│       ├── camera
│       ├── core
│       ├── detector
│       ├── solver
│       └── utils
├── main.cpp
├── README.md
├── src
│   ├── camera
│   ├── core
│   ├── detector
│   ├── solver
│   └── utils
├── test
└── tools
```

## 编译

```bash
mkdir build && cd build
cmake ..
make
./bin/sentry_aim_26
```

## 后续建议

- 添加 camera、detector、solver、serial 模块
- 加入线程管理与状态机
- 引入 OpenVINO 推理接口
- 串口控制功能

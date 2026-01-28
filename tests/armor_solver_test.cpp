/**
 ************************************************************************
 *
 * @file armor_solver_test.cpp
 * @author Neomelt
 * @brief 装甲板位姿求解器测试程序
 *
 ************************************************************************
 * @copyright Copyright (c) 2026 Aurora Vision
 ************************************************************************
 */

#include "auto_aim/armor_solver/solver.hpp"
#include "auto_aim/armor_detector/onnxruntime_detector.hpp"
#include "utils/logger/logger.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <fstream>
#include <sstream>

// 四元数转欧拉角（ZYX顺序，即Yaw-Pitch-Roll）
Eigen::Vector3d quaternionToEulerZYX(const Eigen::Quaterniond& q) {
    Eigen::Vector3d euler;
    
    // Roll (X轴旋转)
    double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
    euler.x() = std::atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (Y轴旋转)
    double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1)
        euler.y() = std::copysign(M_PI / 2, sinp);
    else
        euler.y() = std::asin(sinp);
    
    // Yaw (Z轴旋转)
    double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    euler.z() = std::atan2(siny_cosp, cosy_cosp);
    
    // 转换为角度
    euler = euler * 180.0 / M_PI;
    return euler;
}

// 相机到云台的外参（从配置文件）
const Eigen::Matrix3d R_camera2gimbal = (Eigen::Matrix3d() << 
    -0.0083195760046954614, 0.010498791137270739, 0.99991027599468041,
    -0.99960756138647755, -0.026835747568381807, -0.0080352891314148939,
    0.026748978935305992, -0.99958472279097077, 0.010717933047771133
).finished();

const Eigen::Vector3d t_camera2gimbal(0.094969301833534511, 0.095006290298006682, 0.050987066291756609);

// 将装甲板在相机系的姿态转换到云台系
Eigen::Quaterniond transformArmorPoseToGimbal(const Eigen::Quaterniond& q_armor_in_camera) {
    // 装甲板在相机系的旋转矩阵
    Eigen::Matrix3d R_armor_in_camera = q_armor_in_camera.toRotationMatrix();
    
    // 装甲板在云台系的旋转矩阵 = R_camera2gimbal * R_armor_in_camera
    Eigen::Matrix3d R_armor_in_gimbal = R_camera2gimbal * R_armor_in_camera;
    
    // 转换为四元数
    return Eigen::Quaterniond(R_armor_in_gimbal);
}

// 陀螺仪数据结构
struct GyroData {
    double timestamp;  // 时间戳（毫秒）
    Eigen::Quaterniond quat;  // 四元数 (w, x, y, z)
};

// 从文件加载陀螺仪数据
std::vector<GyroData> loadGyroData(const std::string& filename) {
    std::vector<GyroData> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        utils::logger()->error("[Test] 无法打开陀螺仪数据文件: {}", filename);
        return data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        GyroData gyro;
        double w, x, y, z;
        
        if (iss >> gyro.timestamp >> w >> x >> y >> z) {
            // 陀螺仪四元数格式：w, x, y, z
            gyro.quat = Eigen::Quaterniond(w, x, y, z);
            gyro.quat.normalize();  // 归一化
            data.push_back(gyro);
        }
    }
    
    utils::logger()->info("[Test] 加载了 {} 条陀螺仪数据", data.size());
    return data;
}

// 绘制坐标系辅助函数
void drawAxis(cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, 
              const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs, float length = 0.1f) {
    // 定义坐标系的3个轴（单位：米）
    std::vector<cv::Point3f> axis_points = {
        {0, 0, 0},          // 原点
        {length, 0, 0},     // X轴 (红色)
        {0, length, 0},     // Y轴 (绿色)
        {0, 0, length}      // Z轴 (蓝色)
    };
    
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
    
    // 绘制坐标轴
    cv::line(img, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 3);   // X轴 红色
    cv::line(img, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 3);   // Y轴 绿色
    cv::line(img, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 3);   // Z轴 蓝色
    
    // 在轴端点标注字母
    cv::putText(img, "X", image_points[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(img, "Y", image_points[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, "Z", image_points[3], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
}

// 绘制装甲板信息
void drawArmorInfo(cv::Mat& img, const armor::Armor& armor, const cv::Point2f& center, 
                   const Eigen::Quaterniond* gyro_quat = nullptr) {
    std::stringstream ss;
    int line_offset = -120;  // 增加起始偏移量
    
    // 第1行：装甲板类型和编号
    ss << armor.type << " " << armor.number;
    cv::putText(img, ss.str(), center + cv::Point2f(10, line_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    line_offset += 20;
    
    ss.str("");
    ss.clear();
    
    // 第2行：距离
    ss << "Dist: " << std::fixed << std::setprecision(2) << armor.pos.norm() << "m";
    cv::putText(img, ss.str(), center + cv::Point2f(10, line_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    line_offset += 20;
    
    ss.str("");
    ss.clear();
    
    // 第3行：位置
    ss << "Pos: (" << std::fixed << std::setprecision(2) 
       << armor.pos.x() << "," << armor.pos.y() << "," << armor.pos.z() << ")";
    cv::putText(img, ss.str(), center + cv::Point2f(10, line_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    line_offset += 20;
    
    // 计算PnP的欧拉角（相机系）
    Eigen::Vector3d pnp_euler_camera = quaternionToEulerZYX(armor.ori);
    
    // 将装甲板姿态转换到云台系
    Eigen::Quaterniond armor_quat_gimbal = transformArmorPoseToGimbal(armor.ori);
    Eigen::Vector3d pnp_euler_gimbal = quaternionToEulerZYX(armor_quat_gimbal);
    
    // 第4行：PnP的Yaw角（相机系）
    ss.str("");
    ss.clear();
    ss << "PnP Yaw: " << std::fixed << std::setprecision(2) << pnp_euler_camera.z() << " deg";
    cv::putText(img, ss.str(), center + cv::Point2f(10, line_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    line_offset += 18;
    
    // 第6行：PnP的Pitch和Roll
    ss.str("");
    ss.clear();
    ss << "  P:" << std::fixed << std::setprecision(1) << pnp_euler_camera.y() 
       << " R:" << pnp_euler_camera.x() << " deg";
    cv::putText(img, ss.str(), center + cv::Point2f(10, line_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 200, 200), 1);
    line_offset += 18;
    
    // 如果有陀螺仪数据，只显示作为参考，不做对比
    if (gyro_quat != nullptr) {
        // 计算陀螺仪的欧拉角
        Eigen::Vector3d gyro_euler = quaternionToEulerZYX(*gyro_quat);
        
        // 显示云台Yaw角（参考值）
        ss.str("");
        ss.clear();
        ss << "Gimbal Yaw: " << std::fixed << std::setprecision(2) << gyro_euler.z() << " deg";
        cv::putText(img, ss.str(), center + cv::Point2f(10, line_offset), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 150, 150), 1);
        line_offset += 16;
    }
}

int main(int argc, char** argv) {
    // 初始化日志
    utils::logger()->info("[Test] 装甲板位姿求解器测试开始");

    // 命令行参数：./armor_solver_test <视频路径> [陀螺仪数据路径] [相机配置路径]
    if (argc < 2) {
        utils::logger()->error("[Test] 用法: {} <视频路径> [陀螺仪数据路径] [相机配置路径]", argv[0]);
        utils::logger()->info("[Test] 示例: {} video/demo.avi video/demo.txt config/camera_info.yaml", argv[0]);
        return -1;
    }

    // 1. 初始化检测器
    armor::OnnxRuntimeDetector detector;
    armor::DetectorParams detector_params;
    detector_params.model_path = "models/0526.onnx";
    detector_params.input_size = {640, 640};
    detector_params.conf_threshold = 0.5;
    detector_params.nms_threshold = 0.45;
    detector_params.enable_debug = false;  // 关闭检测器自带的debug，我们自己绘制
    detector.setParams(detector_params);

    // 2. 初始化位姿求解器
    std::string camera_config_path = "config/camera_info.yaml";
    if (argc > 3) {
        camera_config_path = argv[3];
    }
    armor::PnpSolver solver(camera_config_path);
    
    // 加载相机参数用于可视化
    cv::Mat camera_matrix, dist_coeffs;
    try {
        YAML::Node config = YAML::LoadFile(camera_config_path);
        
        // 读取camera_matrix
        auto cm_node = config["camera_matrix"];
        int cm_rows = cm_node["rows"].as<int>();
        int cm_cols = cm_node["cols"].as<int>();
        auto cm_data = cm_node["data"].as<std::vector<double>>();
        
        camera_matrix = cv::Mat(cm_rows, cm_cols, CV_64F);
        for (int i = 0; i < cm_rows; i++) {
            for (int j = 0; j < cm_cols; j++) {
                camera_matrix.at<double>(i, j) = cm_data[i * cm_cols + j];
            }
        }
        
        // 读取distortion_coefficients
        auto dc_node = config["distortion_coefficients"];
        int dc_rows = dc_node["rows"].as<int>();
        int dc_cols = dc_node["cols"].as<int>();
        auto dc_data = dc_node["data"].as<std::vector<double>>();
        
        dist_coeffs = cv::Mat(dc_rows, dc_cols, CV_64F);
        for (int i = 0; i < dc_rows; i++) {
            for (int j = 0; j < dc_cols; j++) {
                dist_coeffs.at<double>(i, j) = dc_data[i * dc_cols + j];
            }
        }
    } catch (const std::exception& e) {
        utils::logger()->error("[Test] 加载相机配置失败: {}", e.what());
        return -1;
    }
    
    if (camera_matrix.empty() || dist_coeffs.empty()) {
        utils::logger()->error("[Test] 相机参数为空");
        return -1;
    }

    // 3. 加载陀螺仪数据（可选）
    std::vector<GyroData> gyro_data;
    bool has_gyro_data = false;
    if (argc > 2) {
        gyro_data = loadGyroData(argv[2]);
        has_gyro_data = !gyro_data.empty();
        if (has_gyro_data) {
            utils::logger()->info("[Test] 陀螺仪数据已加载，将进行对比分析");
        }
    }

    // 4. 打开视频
    std::string video_path = argv[1];
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        utils::logger()->error("[Test] 无法打开视频: {}", video_path);
        return -1;
    }

    // 获取视频FPS用于时间戳计算
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    utils::logger()->info("[Test] 视频已打开: {}, FPS: {:.2f}", video_path, video_fps);
    utils::logger()->info("[Test] 按 'q' 退出, 'p' 暂停/继续");

    cv::Mat frame;
    bool paused = false;
    int frame_count = 0;
    
    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) {
                utils::logger()->info("[Test] 视频播放完毕");
                break;
            }
            frame_count++;
        }

        // 根据帧号查找对应的陀螺仪数据
        const GyroData* current_gyro = nullptr;
        if (has_gyro_data && frame_count > 0 && frame_count <= gyro_data.size()) {
            current_gyro = &gyro_data[frame_count - 1];
        }

        // 5. 检测装甲板
        auto detect_start = std::chrono::high_resolution_clock::now();
        auto armor_objects = detector.detect(frame);
        auto detect_end = std::chrono::high_resolution_clock::now();
        auto detect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

        // 6. 位姿求解
        std::vector<armor::Armor> armors;
        auto solve_start = std::chrono::high_resolution_clock::now();
        for (const auto& obj : armor_objects) {
            armor::Armor armor;
            if (solver.solve(obj, armor)) {
                armors.push_back(armor);
            }
        }
        auto solve_end = std::chrono::high_resolution_clock::now();
        auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start).count();

        // 7. 可视化
        cv::Mat display = frame.clone();
        
        // 绘制每个装甲板
        for (size_t i = 0; i < armor_objects.size() && i < armors.size(); i++) {
            const auto& obj = armor_objects[i];
            const auto& armor = armors[i];
            
            // 绘制四点边框和索引号
            for (int j = 0; j < 4; j++) {
                cv::line(display, obj.pts[j], obj.pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
                cv::circle(display, obj.pts[j], 5, cv::Scalar(255, 0, 255), -1);
                // 显示点的索引号
                cv::putText(display, std::to_string(j), obj.pts[j] + cv::Point2f(-10, -10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            }
            
            // 转换旋转和平移用于绘制
            cv::Mat rvec, tvec;
            Eigen::Matrix3d rot = armor.ori.toRotationMatrix();
            cv::Mat rot_mat = (cv::Mat_<double>(3, 3) << 
                rot(0, 0), rot(0, 1), rot(0, 2),
                rot(1, 0), rot(1, 1), rot(1, 2),
                rot(2, 0), rot(2, 1), rot(2, 2));
            cv::Rodrigues(rot_mat, rvec);
            tvec = (cv::Mat_<double>(3, 1) << armor.pos.x(), armor.pos.y(), armor.pos.z());
            
            // 绘制坐标系
            drawAxis(display, rvec, tvec, camera_matrix, dist_coeffs, 0.05f);
            
            // 绘制信息文本（包含陀螺仪对比）
            const Eigen::Quaterniond* gyro_quat = has_gyro_data && current_gyro ? &current_gyro->quat : nullptr;
            drawArmorInfo(display, armor, obj.center, gyro_quat);
        }
        
        // 绘制统计信息
        std::stringstream info;
        info << "Frame: " << frame_count 
             << " | Detected: " << armor_objects.size() 
             << " | Solved: " << armors.size();
        if (has_gyro_data && current_gyro) {
            info << " | Gyro: (" << std::fixed << std::setprecision(3) 
                 << current_gyro->quat.w() << "," 
                 << current_gyro->quat.x() << "," 
                 << current_gyro->quat.y() << "," 
                 << current_gyro->quat.z() << ")";
        }
        cv::putText(display, info.str(), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        info.str("");
        info << "Detect: " << detect_duration << "ms | Solve: " << solve_duration << "ms";
        cv::putText(display, info.str(), cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        if (paused) {
            cv::putText(display, "PAUSED", cv::Point(10, 90), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Armor Solver Test", display);

        // 按键控制
        char key = static_cast<char>(cv::waitKey(paused ? 0 : 1));
        if (key == 'q' || key == 27) {  // q 或 ESC 退出
            break;
        } else if (key == 'p' || key == ' ') {  // p 或 空格 暂停
            paused = !paused;
            utils::logger()->info("[Test] {}", paused ? "暂停" : "继续");
        }
    }

    utils::logger()->info("[Test] 测试完成");
    return 0;
}

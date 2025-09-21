#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 创建一个简单的图像
    cv::Mat test_image = cv::Mat::zeros(300, 400, CV_8UC3);
    cv::putText(test_image, "OpenCV Display Test", cv::Point(50, 150), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    
    std::cout << "尝试显示测试图像..." << std::endl;
    
    // 尝试显示图像
    cv::imshow("Test Window", test_image);
    
    std::cout << "等待按键..." << std::endl;
    int key = cv::waitKey(3000);  // 等待3秒或按键
    
    std::cout << "按键值: " << key << std::endl;
    
    cv::destroyAllWindows();
    
    return 0;
}

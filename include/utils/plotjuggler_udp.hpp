#pragma once

#include <iomanip>
#include <map>
#include <sstream>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace utils {

/**
 * @brief PlotJuggler UDP 实时流传输类 (Header-only)
 */
class PJStreamer {
public:
    /**
     * @param ip PlotJuggler 运行的 IP，默认本地
     * @param port PlotJuggler UDP Server 监听的端口
     */
    explicit PJStreamer(const std::string& ip = "127.0.0.1", int port = 9870) {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(port);
        addr_.sin_addr.s_addr = inet_addr(ip.c_str());
    }

    ~PJStreamer() {
        if (sock_ >= 0)
            close(sock_);
    }

    // 禁止拷贝，防止 socket 重复关闭
    PJStreamer(const PJStreamer&) = delete;
    PJStreamer& operator=(const PJStreamer&) = delete;

    /**
     * @brief 发送原始 JSON 字符串
     */
    void send_raw(const std::string& json_data) const {
        sendto(sock_, json_data.c_str(), json_data.size(), 0, (struct sockaddr*)&addr_,
               sizeof(addr_));
    }

    /**
     * @brief 封装：自动构建并发送 JSON
     * 使用示例: pj.send_map({ {"x", 1.2}, {"y", 0.5}, {"time", t} });
     */
    void send_map(const std::map<std::string, double>& data_map) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(5) << "{";
        for (auto it = data_map.begin(); it != data_map.end(); ++it) {
            oss << "\"" << it->first << "\":" << it->second;
            if (std::next(it) != data_map.end())
                oss << ",";
        }
        oss << "}";
        send_raw(oss.str());
    }

private:
    int sock_;
    struct sockaddr_in addr_{};
};

}  // namespace utils
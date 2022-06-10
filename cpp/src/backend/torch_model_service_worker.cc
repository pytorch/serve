#include "torch_model_service_worker.hh"

namespace torchserve {
TorchModelServiceWorker::TorchModelServiceWorker(
    const std::string& s_type,
    const std::string& s_name,
    const std::string& host_addr,
    const std::string& port_num)
    : sock_type_(s_type) {
    unsigned short socket_family;
    if (s_type == "unix") {
        if (s_name.empty()) {
            LOG(FATAL) << "Wrong arguments passed. No socket name given.";
        }

        std::experimental::filesystem::path s_name_path(s_name);
        if (remove(s_name.c_str()) != 0 && std::experimental::filesystem::exists(s_name_path)) {
            LOG(FATAL) << "socket already in use: " << s_name;
        }
        sock_name_ = s_name;
        socket_family = AF_UNIX;
    } else if (s_type == "tcp") {
        if (host_addr.empty())
            sock_name_ = "127.0.0.1";
        else
            sock_name_ = host_addr;
        if (port_num.empty())
            LOG(FATAL) << "Wrong arguments passed. No socket port given.";
        port_ = htons(stoi(port_num));
        socket_family = AF_INET;
    } else {
        LOG(FATAL) << "Incomplete data provided";
    }

    LOG(INFO) << "Listening on port: " << s_name;
    sock_ = socket(socket_family, SOCK_STREAM, 0);
    if (sock_ == -1) {
        LOG(FATAL) << "Failed to create socket descriptor. errno: " << errno;
    }
}

TorchModelServiceWorker::~TorchModelServiceWorker() {
    // TODO: Do socket cleanup
};

[[noreturn]] void TorchModelServiceWorker::HandleConnection(int cl_socket) {
    LOG(INFO) << "Handle connection";
    while (true) {
        auto resp = torchserve::RetrieveMsg(cl_socket);
        char cmd = resp.first;
        void* msg = resp.second;
        if (cmd == 'I') {
            LOG(INFO) << "INFER request received";
            //resp = service.predict(msg)
            //cl_socket.sendall(resp)
        } else if (cmd == 'L') {
            LOG(INFO) << "LOAD request received";
            //service, result, code = self.load_model(msg)
            //resp = bytearray()
            //resp += create_load_model_response(code, result)
            //cl_socket.sendall(resp)
            //if code != 200:
            //  raise RuntimeError("{} - {}".format(code, result))
        } else {
            LOG(ERROR) << "Received unknown command: " << cmd;
        }
    }
}

[[noreturn]] void TorchModelServiceWorker::RunServer() {
    // TODO: Add sock accept timeout
    int on = 1;
    setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr* srv_sock_address, client_sock_address{};
    if (sock_type_ == "unix") {
        LOG(INFO) << "Binding to unix socket";
        sockaddr_un sock_addr{};
        std::memset(&sock_addr, 0, sizeof(sock_addr));
        sock_addr.sun_family = AF_UNIX;
        strcpy(sock_addr.sun_path, sock_name_.c_str());
        // TODO: Fix truncation of socket name to 14 chars when casting
        srv_sock_address = reinterpret_cast<sockaddr*>(&sock_addr);
    } else {
        LOG(INFO) << "Binding to udp socket";
        sockaddr_in sock_addr{};
        std::memset(&sock_addr, 0, sizeof(sock_addr));
        sock_addr.sin_family = AF_INET;
        sock_addr.sin_port = port_;
        sock_addr.sin_addr.s_addr = inet_addr(sock_name_.c_str());
        srv_sock_address = reinterpret_cast<sockaddr*>(&sock_addr);
    }

    if (bind(sock_, srv_sock_address, sizeof(*srv_sock_address)) < 0) {
        close(sock_);
        LOG(FATAL) << "Could not bind socket. errno: " << errno;
    }
    listen(sock_, 1);
    LOG(INFO) << "Socket bind successful";
    LOG(INFO) << "[PID]" << getpid();
    LOG(INFO) << "Torch worker started.";

    for (;;) {
        socklen_t len = sizeof(client_sock_address);
        auto client_sock = accept(sock_, (sockaddr *)&client_sock_address, &len);
        if (client_sock < 0)
            LOG(ERROR) << "Failed listening on socket. errno: " << errno;
        LOG(INFO) << "Connection accepted: " << sock_name_;
        HandleConnection(client_sock);
    }
}
} // namespace torchserve

#include "src/backends/process/model_worker.hh"
#include "src/backends/torch_scripted/torch_scripted_backend.hh"

namespace torchserve {
  void SocketServer::Initialize(
    const std::string& socket_type,
    const std::string& socket_name,
    const std::string& host_addr,
    const std::string& port_num,
    const torchserve::RuntimeType& runtime_type,
    torchserve::DeviceType device_type) {
    unsigned short socket_family;
    socket_family = AF_INET;
    if (socket_type == "unix") {
      socket_family = AF_UNIX;
      if (socket_name.empty()) {
        LOG(FATAL) << "Wrong arguments passed. No socket name given.";
      }
      
      std::filesystem::path s_name_path(socket_name);
      if (std::remove(socket_name.c_str()) != 0 && std::filesystem::exists(s_name_path)) {
        LOG(FATAL) << "socket already in use: " << socket_name;
      }
      socket_name_ = socket_name;
    } else if (socket_type == "tcp") {
      if (host_addr.empty()) {
        socket_name_ = "127.0.0.1";
      } else {
        socket_name_ = host_addr;
        if (port_num.empty())
            LOG(FATAL) << "Wrong arguments passed. No socket port given.";
        port_ = htons(stoi(port_num));
      }
    } else {
        LOG(FATAL) << "Incomplete data provided";
    }

    LOG(INFO) << "Listening on port: " << socket_name;
    server_socket_ = socket(socket_family, SOCK_STREAM, 0);
    if (server_socket_ == -1) {
        LOG(FATAL) << "Failed to create socket descriptor. errno: " << errno;
    }
    backend_ = CreateBackend(runtime_type, device_type);
  }

  void SocketServer::Run() {
    // TODO: Add sock accept timeout
    int on = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) == -1) {
      LOG(FATAL) << "Failed to setsockopt. errno: " << errno;
    }

    sockaddr* srv_sock_address, client_sock_address{};
    if (socket_type_ == "unix") {
      LOG(INFO) << "Binding to unix socket";
      sockaddr_un sock_addr{};
      std::memset(&sock_addr, 0, sizeof(sock_addr));
      sock_addr.sun_family = AF_UNIX;
      std::strcpy(sock_addr.sun_path, socket_name_.c_str());
      // TODO: Fix truncation of socket name to 14 chars when casting
      srv_sock_address = reinterpret_cast<sockaddr*>(&sock_addr);
    } else {
      LOG(INFO) << "Binding to udp socket";
      sockaddr_in sock_addr{};
      std::memset(&sock_addr, 0, sizeof(sock_addr));
      sock_addr.sin_family = AF_INET;
      sock_addr.sin_port = port_;
      sock_addr.sin_addr.s_addr = inet_addr(socket_name_.c_str());
      srv_sock_address = reinterpret_cast<sockaddr*>(&sock_addr);
    }

    if (bind(server_socket_, srv_sock_address, sizeof(*srv_sock_address)) < 0) {
      LOG(FATAL) << "Could not bind socket. errno: " << errno;
    }
    if (listen(server_socket_, 1) == -1) {
      LOG(FATAL) << "Failed to listen on socket. errno: " << errno;
    }
    LOG(INFO) << "Socket bind successful";
    LOG(INFO) << "[PID]" << getpid();
    LOG(INFO) << "Torchserve worker started.";

    while (true) {
      socklen_t len = sizeof(client_sock_address);
      auto client_sock = accept(server_socket_, (sockaddr *)&client_sock_address, &len);
      if (client_sock < 0) {
          LOG(FATAL) << "Failed to accept client. errno: " << errno;
      }
      LOG(INFO) << "Connection accepted: " << socket_name_;
      auto model_worker = std::make_unique<torchserve::SocketModelWorker>(client_sock, backend_);
      model_worker->Run();
    }
  }

  std::shared_ptr<torchserve::Backend> SocketServer::CreateBackend(
    const torchserve::RuntimeType& runtime_type,
    torchserve::DeviceType device_type) {
    /**
     * @brief 
     * TODO: Dynamically create backend from the corresponding shared lib 
     * by calling GetBackendLibPath
     */
    if (runtime_type == "LDP") {
      return std::make_shared<TorchScriptedBackend>();
    }
    return nullptr;
  }

/*
    char cmd = data[0];
    OTFMessage::RequestMessage msg;
    if (cmd == LOAD_MSG) {
      msg.load_request = RetrieveLoadMsg(conn);
    } else if (cmd == PREDICT_MSG) {
      //TODO: call msg = RetrieveInferenceMsg(conn);
      std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      LOG(INFO) << "Backend received inference at: " << std::ctime(&end_time);
    } else {
      LOG(ERROR) << "Invalid command: " << cmd;
    }
    return std::make_pair(cmd, msg);
  } 
  */
  [[noreturn]] void SocketModelWorker::Run() {
    LOG(INFO) << "Handle connection";
    while (true) {
      char cmd  = torchserve::OTFMessage::RetrieveCmd(client_socket_);
    
      if (cmd == 'I') {
        LOG(INFO) << "INFER request received";
        // TODO: impl.
      } else if (cmd == 'L') {
        LOG(INFO) << "LOAD request received";
        // TODO: error handling
        auto response = backend_->LoadModel(torchserve::OTFMessage::RetrieveLoadMsg(client_socket_));
        model_instance_ = response.second;
      } else {
        LOG(ERROR) << "Received unknown command: " << cmd;
      }
    }
  }
}

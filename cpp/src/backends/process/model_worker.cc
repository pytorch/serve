#include "src/backends/process/model_worker.hh"

namespace fs = std::filesystem;

namespace torchserve {
  void SocketServer::Initialize(
    const std::string& socket_type,
    const std::string& socket_name,
    const std::string& host_addr,
    const std::string& port_num,
    const torchserve::Manifest::RuntimeType& runtime_type,
    torchserve::DeviceType device_type,
    const std::string& model_dir) {
    unsigned short socket_family;
    socket_family = AF_INET;
    socket_type_ = socket_type;
    if (socket_type == "unix") {
      socket_family = AF_UNIX;
      if (socket_name.empty()) {
        TS_LOG(FATAL, "Wrong arguments passed. No socket name given.");
      }

      fs::path s_name_path(socket_name);
      if (std::remove(socket_name.c_str()) != 0 && fs::exists(s_name_path)) {
        TS_LOGF(FATAL, "socket already in use: {}", socket_name);
      }
      socket_name_ = socket_name;
    } else if (socket_type == "tcp") {
      if (host_addr.empty()) {
        socket_name_ = "127.0.0.1";
      } else {
        socket_name_ = host_addr;
        if (port_num.empty()) {
          TS_LOG(FATAL, "Wrong arguments passed. No socket port given.");
        }
        port_ = htons(stoi(port_num));
      }
    } else {
      TS_LOG(FATAL, "Incomplete data provided");
    }

    TS_LOGF(INFO, "Listening on port: {}", socket_name);
    server_socket_ = socket(socket_family, SOCK_STREAM, 0);
    if (server_socket_ == -1) {
      TS_LOGF(FATAL, "Failed to create socket descriptor. errno: {}", errno);
    }

    if (!CreateBackend(runtime_type, model_dir)) {
      TS_LOGF(FATAL, "Failed to create backend, model_dir: {}", model_dir);
    }
  }

  void SocketServer::Run() {
    // TODO: Add sock accept timeout
    int on = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) == -1) {
      TS_LOGF(FATAL, "Failed to setsockopt. errno: {}", errno);
    }

    sockaddr* srv_sock_address, client_sock_address{};
    socklen_t name_len;
    if (socket_type_ == "unix") {
      TS_LOG(INFO, "Binding to unix socket");
      sockaddr_un sock_addr{};
      std::memset(&sock_addr, 0, sizeof(sock_addr));
      sock_addr.sun_family = AF_UNIX;
      std::strcpy(sock_addr.sun_path, socket_name_.c_str());
      srv_sock_address = reinterpret_cast<sockaddr*>(&sock_addr);
      name_len = SUN_LEN(&sock_addr);
    } else {
      TS_LOG(INFO, "Binding to tcp socket");
      sockaddr_in sock_addr{};
      std::memset(&sock_addr, 0, sizeof(sock_addr));
      sock_addr.sin_family = AF_INET;
      sock_addr.sin_port = port_;
      sock_addr.sin_addr.s_addr = inet_addr(socket_name_.c_str());
      srv_sock_address = reinterpret_cast<sockaddr*>(&sock_addr);
      name_len = sizeof(*srv_sock_address);
    }

    if (bind(server_socket_, srv_sock_address, name_len) < 0) {
      TS_LOGF(FATAL, "Could not bind socket. errno: {}", errno);
    }
    if (listen(server_socket_, 1) == -1) {
      TS_LOGF(FATAL, "Failed to listen on socket. errno: {}", errno);
    }
    TS_LOG(INFO, "Socket bind successful");
    TS_LOGF(INFO, "[PID]{}", getpid());
    TS_LOG(INFO, "Torch worker started.");

    while (true) {
      socklen_t len = sizeof(client_sock_address);
      auto client_sock = accept(server_socket_, (sockaddr *)&client_sock_address, &len);
      if (client_sock < 0) {
        TS_LOGF(FATAL, "Failed to accept client. errno: {}", errno);
      }
      TS_LOGF(INFO, "Connection accepted: {}", socket_name_);
      auto model_worker = std::make_unique<torchserve::SocketModelWorker>(client_sock, backend_);
      model_worker->Run();
    }
  }

  bool SocketServer::CreateBackend(
    const torchserve::Manifest::RuntimeType& runtime_type,
    const std::string& model_dir) {
    if (runtime_type == "LSP") {
      backend_ = std::make_shared<torchserve::torchscripted::Backend>();
      return backend_->Initialize(model_dir);
    }
    return false;
  }

  [[noreturn]] void SocketModelWorker::Run() {
    TS_LOG(INFO, "Handle connection");
    while (true) {
      char cmd  = torchserve::OTFMessage::RetrieveCmd(client_socket_);

      if (cmd == 'I') {
        TS_LOG(INFO, "INFER request received");
        auto model_instance = backend_->GetModelInstance();
        if (!model_instance) {
          TS_LOG(ERROR, "Model is not loaded yet, not able to process this inference request.");
        } else {
          auto response = model_instance->Predict(torchserve::OTFMessage::RetrieveInferenceMsg(client_socket_));
          if(!torchserve::OTFMessage::SendInferenceResponse(client_socket_, response)) {
            TS_LOG(ERROR, "Error writing inference response to socket");
         }
        }
      } else if (cmd == 'L') {
        TS_LOG(INFO, "LOAD request received");
        // TODO: error handling
        auto backend_response = backend_->LoadModel(torchserve::OTFMessage::RetrieveLoadMsg(client_socket_));
        if (!torchserve::OTFMessage::SendLoadModelResponse(client_socket_, std::move(backend_response))) {
          TS_LOG(ERROR, "Error writing response to socket");
        }
      } else {
        TS_LOGF(ERROR, "Received unknown command: {}", cmd);
      }
    }
  }
} // namespace torchserve

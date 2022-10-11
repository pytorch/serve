#ifndef TS_CPP_BACKENDS_PROCESS_MODEL_WORKER_HH_
#define TS_CPP_BACKENDS_PROCESS_MODEL_WORKER_HH_

#include <arpa/inet.h>
#include <cstdio>
#include <filesystem>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <tuple>
#include <unistd.h>
#include <memory>

#include "src/backends/core/backend.hh"
#include "src/backends/protocol/otf_message.hh"
#include "src/backends/torch_scripted/torch_scripted_backend.hh"
#include "src/utils/config.hh"
#include "src/utils/logging.hh"
#include "src/utils/model_archive.hh"

namespace torchserve {
  class SocketServer {
    public:
    static SocketServer GetInstance() {
      static SocketServer instance;
      return instance;
    };
    ~SocketServer() {
       if (server_socket_ >= 0) {
          close(server_socket_);
       }
     };

    void Initialize(
      const std::string& socket_type,
      const std::string& socket_name,
      const std::string& host_addr,
      const std::string& port_num,
      const torchserve::Manifest::RuntimeType& runtime_type,
      torchserve::DeviceType device_type,
      const std::string& model_dir);
    
    void Run();

    private:
    SocketServer() {};
    bool CreateBackend(
      const torchserve::Manifest::RuntimeType& runtime_type,
      const std::string& model_dir);

    // TODO; impl.
    //short MAX_FAILURE_THRESHOLD = 5;
    //float SOCKET_ACCEPT_TIMEOUT = 30.0f;
    int server_socket_;
    std::string socket_type_;
    std::string socket_name_;
    ushort port_;
    std::shared_ptr<torchserve::Backend> backend_;
  };

  class SocketModelWorker {
    public:
    SocketModelWorker(int client_socket, std::shared_ptr<torchserve::Backend> backend) : 
    client_socket_(client_socket), backend_(backend) {};
    ~SocketModelWorker() = default;

    [[noreturn]] void Run();

    private:
    Socket client_socket_;
    std::shared_ptr<torchserve::Backend> backend_;
  };
} // namespace torchserve
#endif // TS_CPP_BACKENDS_PROCESS_MODEL_WORKER_HH_
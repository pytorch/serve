#ifndef TS_CPP_BACKENDS_PROCESS_MODEL_WORKER_HH_
#define TS_CPP_BACKENDS_PROCESS_MODEL_WORKER_HH_

#include <arpa/inet.h>
#include <cstdio>
#include <filesystem>
#include <glog/logging.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <tuple>
#include <unistd.h>
#include <memory>

#include "src/backends/core/backend.hh"
#include "src/backends/protocol/otf_message.hh"
#include "src/utils/config.hh"
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
      const torchserve::RuntimeType& runtime_type,
      torchserve::DeviceType device_type);
    
    void Run();

    private:
    SocketServer() {};
    std::shared_ptr<torchserve::Backend> CreateBackend(
      const torchserve::RuntimeType& runtime_type,
      torchserve::DeviceType device_type);

    // TODO; impl.
    //short MAX_FAILURE_THRESHOLD = 5;
    //float SOCKET_ACCEPT_TIMEOUT = 30.0f;
    Socket server_socket_;
    std::string socket_type_;
    std::string socket_name_;
    ushort port_;
    std::shared_ptr<torchserve::Backend> backend_;
  };

  class SocketModelWorker {
    public:
    SocketModelWorker(Socket client_socket, std::shared_ptr<torchserve::Backend> backend) : 
    client_socket_(client_socket), backend_(backend) {};
    ~SocketModelWorker() {
      if (client_socket_ >= 0) {
        close(client_socket_);
      }
    };

    [[noreturn]] void Run();

    private:
    Socket client_socket_;
    std::shared_ptr<torchserve::ModelInstance> model_instance_;
    std::shared_ptr<torchserve::Backend> backend_;
  };
} // namespace torchserve
#endif // TS_CPP_BACKENDS_PROCESS_MODEL_WORKER_HH_
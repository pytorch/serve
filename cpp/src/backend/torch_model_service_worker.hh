#pragma once

#ifndef TS_CPP_BACKEND_TORCH_MODEL_SERVICE_WORKER_H
#define TS_CPP_BACKEND_TORCH_MODEL_SERVICE_WORKER_H

#include <arpa/inet.h>
#include <cstdio>
#include <experimental/filesystem>
#include <glog/logging.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <tuple>
#include <unistd.h>
#include "proc/protocol.hh"

namespace torchserve {

class TorchModelServiceWorker {
 public:
  TorchModelServiceWorker(
      const std::string& s_type,
      const std::string& s_name,
      const std::string& host_addr,
      const std::string& port_num);
  ~TorchModelServiceWorker();

  [[noreturn]] void RunServer();

 private:
  short MAX_FAILURE_THRESHOLD = 5;
  float SOCKET_ACCEPT_TIMEOUT = 30.0f;
  std::string sock_type_;
  std::string sock_name_;
  ushort port_;
  int sock_;
  // LoadModel(load_model_request);
  [[noreturn]] static void HandleConnection(int cl_socket);
};

} // namespace torchserve


#endif //TS_CPP_BACKEND_TORCH_MODEL_SERVICE_WORKER_H
#include <iostream>
#include <memory>
#include <gflags/gflags.h>

#include "src/utils/logging.hh"
#include "src/backends/process/model_worker.hh"

DEFINE_string(socket_type, "tcp", "socket type");
DEFINE_string(socket_name, "", "socket name for uds");
DEFINE_string(host, "127.0.0.1", "");
DEFINE_string(port, "9000", "");
DEFINE_string(runtime_type, "LSP", "model runtime type");
DEFINE_string(device_type, "cpu", "cpu, or gpu");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Init logging
  google::InitGoogleLogging("ts_cpp_backend");
  FLAGS_logtostderr = 1;
  // TODO: Set logging format same as python worker
  LOG(INFO) << "Initializing Libtorch backend worker...";

  torchserve::SocketServer server = torchserve::SocketServer::GetInstance();

  server.Initialize(FLAGS_socket_type, FLAGS_socket_name, 
  FLAGS_host, FLAGS_port, FLAGS_runtime_type, FLAGS_device_type);

  server.Run();

  gflags::ShutDownCommandLineFlags();
}
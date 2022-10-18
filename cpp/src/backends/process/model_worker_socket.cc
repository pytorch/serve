#include <gflags/gflags.h>

#include <iostream>
#include <memory>

#include "src/backends/process/model_worker.hh"
#include "src/utils/logging.hh"

DEFINE_string(socket_type, "tcp", "socket type");
DEFINE_string(socket_name, "", "socket name for uds");
DEFINE_string(host, "127.0.0.1", "");
DEFINE_string(port, "9000", "");
DEFINE_string(runtime_type, "LSP", "model runtime type");
DEFINE_string(device_type, "cpu", "cpu, or gpu");
// TODO: discuss multiple backends support
DEFINE_string(model_dir, "", "model path");
// TODO: change to file based config
DEFINE_string(logger_config_path, "./_build/resources/logging.config",
              "Logging config file path");

int main(int argc, char* argv[]) {
  try {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    torchserve::Logger::InitLogger(FLAGS_logger_config_path);

    torchserve::SocketServer server = torchserve::SocketServer::GetInstance();
    server.Initialize(FLAGS_socket_type, FLAGS_socket_name, FLAGS_host,
                      FLAGS_port, FLAGS_runtime_type, FLAGS_device_type,
                      FLAGS_model_dir);

    server.Run();

    gflags::ShutDownCommandLineFlags();
  } catch (...) {
    std::cout << "cpp backend failed to start\n";
  }
}
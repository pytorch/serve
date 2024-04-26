#include <gflags/gflags.h>

#include <filesystem>
#include <iostream>
#include <memory>

#include "src/backends/process/model_worker.hh"
#include "src/utils/logging.hh"
#include "src/utils/metrics/registry.hh"

DEFINE_string(sock_type, "tcp", "socket type");
DEFINE_string(sock_name, "", "socket name for uds");
DEFINE_string(host, "127.0.0.1", "");
DEFINE_string(port, "9000", "");
DEFINE_string(runtime_type, "LSP", "model runtime type");
DEFINE_string(device_type, "cpu", "cpu, or gpu");
// TODO: discuss multiple backends support
DEFINE_string(model_dir, "", "model path");
DEFINE_string(logger_config_path, "", "Logging config file path");
DEFINE_string(metrics_config_path, "", "Metrics config file path");

int main(int argc, char* argv[]) {
  try {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    torchserve::Logger::InitLogger(FLAGS_logger_config_path);

    if (FLAGS_metrics_config_path.empty()) {
      FLAGS_metrics_config_path =
          std::string() +
          std::filesystem::canonical(gflags::ProgramInvocationName())
              .parent_path()
              .c_str() +
          PATH_SEPARATOR + ".." + PATH_SEPARATOR + ".." + PATH_SEPARATOR +
          ".." + PATH_SEPARATOR + "ts" + PATH_SEPARATOR + "configs" +
          PATH_SEPARATOR + "metrics.yaml";
    }
    torchserve::MetricsRegistry::Initialize(
        FLAGS_metrics_config_path, torchserve::MetricsContext::BACKEND);

    torchserve::SocketServer server = torchserve::SocketServer::GetInstance();
    server.Initialize(FLAGS_sock_type, FLAGS_sock_name, FLAGS_host, FLAGS_port,
                      FLAGS_runtime_type, FLAGS_device_type, FLAGS_model_dir);

    server.Run();

    gflags::ShutDownCommandLineFlags();
  } catch (...) {
    std::cout << "cpp backend failed to start\n";
  }
}

#include "src/utils/config.hh"

namespace torchserve {
  BackendFrameworkConfig::BackendFrameworkConfig() {
    // TODO: replace 1.11 with default version variable
    runtime_to_framework_map_ = {
      {"LDP", std::make_shared<torchserve::BackendFrameworkConfig::Framework>(
        "torch_deploy", "1.11")},
      {"LSP", std::make_shared<torchserve::BackendFrameworkConfig::Framework>(
        "torch_scripted", "1.11")},
    };
  }

  const std::string BackendFrameworkConfig::GetBackendLibPath(
      const torchserve::Manifest::RuntimeType& runtime_type, 
      DeviceType device_type,
      const std::string& version) {
    auto it = runtime_to_framework_map_.find(runtime_type);
    if (it == runtime_to_framework_map_.end()) {
      throw std::invalid_argument(fmt::format(
        "Invalid backend framework: {}", runtime_type));
    }
    
    // TODO: add lib path prefix and version
    // eg. torchserve/backends/torch_scripted/1.11/libts_backends_torch_scripted.so
    #ifdef __APPLE__
      if (device_type == "gpu") {
        throw std::invalid_argument(fmt::format(
          "Invalid device type: {}", device_type));
      }
      return fmt::format("{}/lib{}_{}_backend.dylib", version, it->second->name, device_type);
    #else 
      return fmt::format("{}/lib{}_{}_backend.so", version, it->second->name, device_type);
    #endif
  }
} //namespace torchserve
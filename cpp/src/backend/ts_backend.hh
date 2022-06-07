#pragma once

#include "src/util/runtime_type.hh"

namespace torchserve {
class TorchServeBackend {
  public:
  TorchServeBackend(const std::string &ts_lib_path, RuntimeType runtimeType);

  virtual ~TorchServeBackend();

  virtual void load_model() = 0;

  virtual void predict(std::shared_ptr<BatchJob> job) = 0;

  void ping();

  void handleConnection(uint32_t port);

  void runServer();

};

class BackendModel {
  public:
  virtual ~BackendModel() = 0;

  virtual void predict(std::shared_ptr<BatchJob> job) = 0;
};
}  // namespace torchserve
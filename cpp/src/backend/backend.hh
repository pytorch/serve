#pragma once

#include <memory>
#include <string>
#include "src/core/job.hh"
#include "src/core/model.hh"

namespace torchserve {
  enum BackendType {
    PYTHON,
    TORCH
  };

class Backend {
  public:
  Backend(const std::string backendName, RuntimeType runtimeType);

  virtual ~Backend();

  virtual BackendModel loadModel(const Model &model, int gpuId) = 0;

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
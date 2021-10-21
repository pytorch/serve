#pragma once

#include <deque>
#include <memory>
#include <string>
#include <folly/executors/thread_factory/ThreadFactory.h>
#include <thread>
#include "src/core/model.hh"
#include "src/core/scheduler.hh"
#include "src/core/status.hh"

namespace torchserve {
// A model instance represents a model copy on CPU or GPU.
class ModelInstance {
  public:
  uint16_t intanceId;

  // Load model on CPU if gpuId = -1.
  virtual Status createInstance(const Model &model, int gpuId) = 0;

  // Predict the input request.
  void predict(std::shared_ptr<BatchJob> &job) = 0;

  // Warmup the model.
  virtual void Warmup() = 0;

  uint16_t GetInstanceId();
};

// A ModelInstanceIMP object represents a model copy residenting in main process.
class ModelInstanceIMP : public ModelInstance {
  public:
  ModelInstanceIMP(uint16_t instanceId);

  Status createInstance(const Model &model, int gpuId);

  void predict(std::shared_ptr<BatchJob> &job);

  void warmup();

  private:
  std::shared_ptr<BackendModel> backendModel;
};

// A ModelInstanceOMP object represents an agent of a model copy residenting in out-of-main-process.
class ModelInstanceOMP : public ModelInstance {
  public:
  ModelInstanceOMP(uint16_t instanceId, uint32_t port);

  // Create remote server and load model
  Status createInstance(const Model &model, int gpuId);

  void predict(std::shared_ptr<BatchJob> &job);

  void warmup();

  uint32_t getRemotePort();
  
  private:
  uint32_t port;
  Status connectBackend(uint32_t port);
};

class Worker {
  public:
  uint16_t workerId;
  
  virtual void run(Job &job) = 0;
};

// A WorkerIMP object is associated with a ModelInstanceIMP object.
class WorkerIMP : public Worker {
  public:
  std::shared_ptr<ModelInstanceIMP> modelInstance;

  WorkerIMP(std::shared_ptr<ModelInstanceIMP> &modelInstance);
  void run(Job &job);
};

// A WorkerOMP is an agent of a backend worker associated with a ModelInstanceOMP.
class WorkerOMP : public Worker {
  public:
  std::shared_ptr<ModelInstanceOMP> modelInstance;

  WorkerOMP(std::shared_ptr<ModelInstanceOMP> &modelInstance);
  void run(Job &job);

  private:
  uint32_t port;
  Status connectBackend(uint32_t port);
};
}  // namespace torchserve
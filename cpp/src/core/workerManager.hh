#pragma once

#include <deque>
#include <memory>
#include <string>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include "blockingconcurrentqueue.h"
#include "src/core/model.hh"
#include "src/core/scheduler.hh"
#include "src/core/status.hh"
#include "src/core/worker.hh"
#include "src/backend/backend.hh"

namespace torchserve {

template <class T1>
class WorkerManager : public Scheduler<T1> {
  public:
  static std::unordered_map<BackendType, std::shared_ptr<Backend>> backendMap;

  ~WorkerManager();

  // Load backend if it is needed.
  Status registerBackend(const Model &model);

  // Create model instances concurrently; store them into modelInstances.  
  // A model instance residents in either in main process or out-of-main-process.
  // A model instance is a wrapper of a model copy on CPU or GPU.
  Status createModelInstances(const Model &model);

  // Create workerThreadPool. Each worker thread associates with one modelInstance.
  Status addWorkers(int numWorkers);

  // Add a job into jobQueue.
  void enqueue(const std::unique_ptr<T1> &job);
  
  private:
  std::shared_ptr<Backend> backend;
  moodycamel::BlockingConcurrentQueue<BatchJob> batchJobQueue;
  std::vector<std::shared_ptr<ModelInstance> > modelInstances;
  std::unique_ptr<folly::CPUThreadPoolExecutor>> workerThreadPool;
};
}  // namespace torchserve
#pragma once

#include <string>
#include <unordered_map>

#include "src/core/job.hh"
#include "src/core/model_archive.hh"
#include "ssrc/core/cheduler.hh"
#include "src/core/status.hh"

namespace torchserve {
struct ModelConfig {
  std::string url;
  // The number of a model's copies on CPU or GPU.
  uint8_t numInstances;
  // The minimum number of workers distributed on this model's instances.
  uint16_t minWorkers;
  // The maximum number of workers distributed on this model's instances.
  uint16_t maxWorkers;
  // The maximum size of queue holding incoming requests for the model.
  uint32_t maxQueueSize;
  // The maximum etry attempts for a model incase of a failure.
  uint16_t maxRetries;
  // The maximum batch size in ms that a model is expected to handle.
  uint16_t batchSize;
  // The maximum batch delay time TorchServe waits to receive batch_size number
  // of requests.
  uint32_t maxBatchDelayMSec;
  // the timeout in second of a model's response.
  uint32_t responseTimeoutSec;
};

// WorkflowSpec defined in
// https://github.com/pytorch/serve/blob/master/docs/workflows.md#workflow-specification-file
struct WorkflowSpec {
  // The configuration for a single model or an ensemble model.
  ModelConfig modelConfig;

  // The mapping b/w a model and its configuration.
  // key: a model version name; value: a model configuration.
  std::unordered_map<std::string, std::unique_ptr<ModelConfig> > subModels;

  // Descripes the workflow of an ensemble model.
  // key: a model version name;
  // value: a list of model version names reached out by the model defined in
  // the key.
  std::unordered_map<std::string, std::vector<std::string> > dag;
};

class Model {
  public:
  WorkflowSpec workflowSpec;

  Model(const Manifest &manifest);

  // Update a model configuration
  Status updateModel(const ModelConfig &modelConfig, bool isStartup,
                     bool isCleanup);
  // Add a job into the model's scheduler queue.
  void addJob(Job &job);
  
  private:
  std::unique_ptr<Manifest> manifest;
  // A chain of schedulers.
  std::unique_ptr<Scheduler> schedulers;
};
}  // namespace torchserve
#pragma once

#include <string>
#include <map>
#include <vector>

#include "src/util/model_archive.hh"

namespace torchserve {
  struct ModelConfig {
    // The model alias name.
    std::string modelName;
    // The path of the model file.
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
    // The timeout in second of a model's response.
    uint32_t responseTimeoutSec;
    // The gpu assignment.
    std::vector<short> gpuIds;
    // The runtime type.
    RuntimeType runtimeType;
    // The model's manifest.
    Manifest manifest;
  };
}  // namespace torchserve
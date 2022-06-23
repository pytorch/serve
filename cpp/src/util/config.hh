#ifndef CPP_UTIL_CONFIG_HH_
#define CPP_UTIL_CONFIG_HH_

#include <string>
#include <map>
#include <vector>

#include "util/model_archive.hh"

namespace torchserve {
  struct ModelConfig {
    // The model alias name.
    std::string model_name;
    // The path of the model file.
    std::string url;
    // The number of a model's copies on CPU or GPU.
    uint8_t num_instances;
    // The minimum number of workers distributed on this model's instances.
    uint16_t min_workers;
    // The maximum number of workers distributed on this model's instances.
    uint16_t max_workers;
    // The maximum size of queue holding incoming requests for the model.
    uint32_t max_queue_size;
    // The maximum etry attempts for a model incase of a failure.
    uint16_t max_retries;
    // The maximum batch size in ms that a model is expected to handle.
    uint16_t batch_size;
    // The maximum batch delay time TorchServe waits to receive batch_size number
    // of requests.
    uint32_t max_batch_delay_msec;
    // The timeout in second of a model's response.
    uint32_t response_timeout_sec;
    // The gpu assignment.
    std::vector<short> gpu_ids;
    // The runtime type.
    RuntimeType runtime_type;
    // The model's manifest.
    Manifest manifest;
  };
}  // namespace torchserve
#pragma once

#include "src/util/config.hh"
#include "src/util/message.hh"

namespace torchserve {
  class TorchServeBackend {
    public:
    TorchServeBackend(torchserve::ModelConfig &modelConfig);

    virtual ~TorchServeBackend();

    torchserve::Status load_model();

    torchserve::InferenceResponse predict(
      const torchserve::InferenceRequest &inferenceRequest);
  };
}  // namespace torchserve
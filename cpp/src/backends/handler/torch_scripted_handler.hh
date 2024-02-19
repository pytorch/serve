#pragma once
#include "base_handler.hh"

namespace torchserve {

class TorchScriptHandler : public BaseHandler {
  std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>> LoadModel(
      std::shared_ptr<LoadModelRequest>& load_model_request) override;
};
}  // namespace torchserve

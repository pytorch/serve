#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <string>

#include "src/backends/core/backend.hh"
#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace torchserve {
namespace torchscripted {
class ModelInstance final : public torchserve::ModelInstance {
 public:
  ModelInstance(
      const std::string& instance_id,
      std::shared_ptr<torch::jit::script::Module> model,
      std::shared_ptr<torchserve::torchscripted::BaseHandler>& handler,
      std::shared_ptr<torch::Device> device)
      : torchserve::ModelInstance(instance_id),
        model_(model),
        handler_(handler),
        device_(device){};
  ~ModelInstance() override = default;

  std::shared_ptr<torchserve::InferenceResponseBatch> Predict(
      std::shared_ptr<torchserve::InferenceRequestBatch> request_batch)
      override;

 private:
  std::shared_ptr<torch::jit::script::Module> model_;
  std::shared_ptr<torchserve::torchscripted::BaseHandler> handler_;
  std::shared_ptr<torch::Device> device_;
};
}  // namespace torchscripted
}  // namespace torchserve

#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <string>

#include "src/backends/handler/base_handler.hh"

namespace torchserve {
class ModelInstance {
 public:
  ModelInstance(const std::string& instance_id, std::shared_ptr<void> model,
                std::shared_ptr<torchserve::BaseHandler>& handler,
                std::shared_ptr<torch::Device> device);
  virtual ~ModelInstance() = default;

  std::shared_ptr<torchserve::InferenceResponseBatch> Predict(
      std::shared_ptr<torchserve::InferenceRequestBatch> request_batch);

 protected:
  // instance_id naming convention:
  // device_type + ":" + device_id (or object id)
  std::string instance_id_;
  std::shared_ptr<void> model_;
  std::shared_ptr<torchserve::BaseHandler> handler_;
  std::shared_ptr<torch::Device> device_;
};
}  // namespace torchserve

#include "model_instance.hh"

#include <memory>

namespace torchserve {

ModelInstance::ModelInstance(const std::string& instance_id,
                             std::shared_ptr<void> model,
                             std::shared_ptr<torchserve::BaseHandler>& handler,
                             std::shared_ptr<torch::Device> device)
    : instance_id_(instance_id),
      model_(model),
      handler_(handler),
      device_(device) {}

std::shared_ptr<torchserve::InferenceResponseBatch> ModelInstance::Predict(
    std::shared_ptr<torchserve::InferenceRequestBatch> request_batch) {
  auto response_batch = std::make_shared<torchserve::InferenceResponseBatch>();
  handler_->Handle(model_, device_, request_batch, response_batch);

  return response_batch;
}

}  // namespace torchserve

#include "src/backends/torch_scripted/model_instance.hh"

#include <memory>

namespace torchserve {
namespace torchscripted {

std::shared_ptr<torchserve::InferenceResponseBatch> ModelInstance::Predict(
    std::shared_ptr<torchserve::InferenceRequestBatch> request_batch) {
  auto response_batch = std::make_shared<torchserve::InferenceResponseBatch>();
  handler_->Handle(model_, device_, request_batch, response_batch);

  return response_batch;
}

}  // namespace torchscripted
}  // namespace torchserve

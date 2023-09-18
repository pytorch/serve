#ifndef RESNET_HANDLER_HH_
#define RESNET_HANDLER_HH_

#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace resnet {
class ResnetHandler : public torchserve::torchscripted::BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  ResnetHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~ResnetHandler() override = default;

  std::vector<torch::jit::IValue> Preprocess(
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;

  void Postprocess(
      const torch::Tensor& data,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;
};
}  // namespace resnet
#endif  // RESNET_HANDLER_HH_

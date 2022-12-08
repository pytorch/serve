#pragma once

#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace text_classifier {
class TextClassifierHandler : public torchserve::torchscripted::BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  TextClassifierHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~TextClassifierHandler() override = default;

  std::vector<torch::jit::IValue> Preprocess(
      std::shared_ptr<torch::Device> &device,
      std::pair<std::string&, std::map<uint8_t, std::string>&> &idx_to_req_id,
      std::shared_ptr<torchserve::InferenceRequestBatch> &request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch)
      override;

  void Postprocess(
          const torch::Tensor &data,
          std::pair<std::string&, std::map<uint8_t, std::string>&> &idx_to_req_id,
          std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch)
          override;
};
}  // namespace text_classifier

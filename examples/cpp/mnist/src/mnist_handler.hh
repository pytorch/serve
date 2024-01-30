#pragma once

#include "src/backends/handler/torch_scripted_handler.hh"

namespace mnist {
class MnistHandler : public torchserve::TorchScriptHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  MnistHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~MnistHandler() override = default;

  void Postprocess(
      c10::IValue& data,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;
};
}  // namespace mnist

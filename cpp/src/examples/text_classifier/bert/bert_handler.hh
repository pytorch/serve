#ifndef BERT_HANDLER_HH_
#define BERT_HANDLER_HH_

#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace bert {
class BertHandler : public torchserve::torchscripted::BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  BertHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~BertHandler() override = default;

  std::vector<torch::jit::IValue> Preprocess(
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;

  torch::Tensor Inference(
      std::shared_ptr<torch::jit::script::Module> model,
      std::vector<torch::jit::IValue>& inputs,
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;

  void Postprocess(
      const torch::Tensor& data,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;
};
}  // namespace bert
#endif  // BERT_HANDLER_HH_

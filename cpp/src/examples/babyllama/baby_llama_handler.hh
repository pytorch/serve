#ifndef LLM_HANDLER_HH_
#define LLM_HANDLER_HH_

#include <folly/FileUtil.h>
#include <folly/json.h>

#include <iostream>

// #include "run.c"
#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace llm {
class BabyLlamaHandler : public torchserve::torchscripted::BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  BabyLlamaHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~BabyLlamaHandler() noexcept;

  void initialize_context();

  virtual std::pair<std::shared_ptr<torch::jit::script::Module>,
                    std::shared_ptr<torch::Device>>
  LoadModel(std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

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
}  // namespace llm
#endif  // LLM_HANDLER_HH_
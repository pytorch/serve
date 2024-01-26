#pragma once

#include <folly/FileUtil.h>
#include <folly/json.h>

#include "common/common.h"
#include "ggml.h"
#include "llama.h"
#include "src/backends/handler/base_handler.hh"

namespace llm {
class LlamaCppHandler : public torchserve::BaseHandler {
 private:
  gpt_params params;
  llama_model_params model_params;
  llama_model* llamamodel;
  llama_context_params ctx_params;
  llama_context* llama_ctx;
  const int max_context_size = 32;

 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  LlamaCppHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~LlamaCppHandler() noexcept;

  void initialize_context();

  virtual std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>>
  LoadModel(std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

  c10::IValue Preprocess(
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;

  c10::IValue Inference(
      std::shared_ptr<void> model, c10::IValue& inputs,
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;

  void Postprocess(
      c10::IValue& data,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)
      override;
};
}  // namespace llm

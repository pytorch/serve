#pragma once

#include <tokenizers_cpp.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "src/backends/handler/base_handler.hh"
#include "src/utils/json.hh"

namespace bert {
class BertCppHandler : public torchserve::BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  BertCppHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~BertCppHandler() noexcept = default;

  std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>> LoadModel(
      std::shared_ptr<torchserve::LoadModelRequest>& load_model_request)
      override;

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

private:
  std::unique_ptr<torchserve::Json> mapping_json_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<YAML::Node> model_config_yaml_;
  int max_length_;
};
}  // namespace bert

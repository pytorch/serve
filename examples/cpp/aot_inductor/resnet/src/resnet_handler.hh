#pragma once

#include <folly/FileUtil.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <fmt/format.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_model_container_runner.h>
#include <torch/csrc/inductor/aoti_model_container_runner_cuda.h>
#include <yaml-cpp/yaml.h>

#include "src/backends/handler/base_handler.hh"

namespace resnet {
class ResnetCppHandler : public torchserve::BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  ResnetCppHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  ~ResnetCppHandler() noexcept = default;

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
  std::unique_ptr<folly::dynamic> LoadJsonFile(const std::string& file_path);
  const folly::dynamic& GetJsonValue(std::unique_ptr<folly::dynamic>& json, const std::string& key);
  std::string MapClassToLabel(const torch::Tensor& classes, const torch::Tensor& probs);

  std::unique_ptr<folly::dynamic> mapping_json_;
  std::unique_ptr<YAML::Node> model_config_yaml_;
};
}  // namespac

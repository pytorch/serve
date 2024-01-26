#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <ratio>
#include <utility>

#include "src/utils/logging.hh"
#include "src/utils/message.hh"
#include "src/utils/metrics/registry.hh"
#include "src/utils/model_archive.hh"

namespace torchserve {
/**
 * @brief
 * TorchBaseHandler <=> BaseHandler:
 * serve/ts/torch_handler/base_handler.py#L37
 *
 * TorchBaseHandler is not responsible for loading model since it is derived
 * from TorchScritpedModelInstance.
 */
class BaseHandler {
 public:
  // NOLINTBEGIN(bugprone-exception-escape)
  BaseHandler() = default;
  // NOLINTEND(bugprone-exception-escape)
  virtual ~BaseHandler() = default;

  virtual void Initialize(const std::string& model_dir,
                          std::shared_ptr<torchserve::Manifest>& manifest) {
    model_dir_ = model_dir;
    manifest_ = manifest;
  };

  virtual std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>>
  LoadModel(std::shared_ptr<LoadModelRequest>& load_model_request) = 0;

  virtual c10::IValue Preprocess(
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);

  virtual c10::IValue Inference(
      std::shared_ptr<void> model, c10::IValue& inputs,
      std::shared_ptr<torch::Device>& device,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);

  virtual void Postprocess(
      c10::IValue& data,
      std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);

  /**
   * @brief
   * function Predict <=> entry point function handle
   * /serve/ts/torch_handler/base_handler.py#L205
   * @param inference_request
   * @return std::shared_ptr<torchserve::InferenceResponse>
   */
  void Handle(
      std::shared_ptr<void> model, std::shared_ptr<torch::Device>& device,
      std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);

 protected:
  std::shared_ptr<torch::Device> GetTorchDevice(
      std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

  std::shared_ptr<torchserve::Manifest> manifest_;
  std::string model_dir_;
};
}  // namespace torchserve

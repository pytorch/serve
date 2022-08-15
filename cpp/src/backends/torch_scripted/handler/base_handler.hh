#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_BASE_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_BASE_HANDLER_HH_

#include <torch/script.h>
#include <torch/torch.h>
#include <functional>
#include <map>
#include <utility>

#include "src/backends/core/backend.hh"

namespace torchserve {
  namespace torchscripted {
    /**
     * @brief 
     * TorchBaseHandler <=> BaseHandler:
     * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L37
     * 
     * TorchBaseHandler is not responsible for loading model since it is derived from
     * TorchScritpedModelInstance.
     */
    class BaseHandler {
      public:
      BaseHandler();
      virtual ~BaseHandler() {};

      virtual void Initialize(
        const std::string& model_path,
        std::shared_ptr<torchserve::Manifest>& manifest) {
        model_path_ = model_path;
        manifest_ = manifest;
      };

      virtual std::pair<std::shared_ptr<torch::jit::script::Module>, torch::Device> 
      LoadModel(
        std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

      virtual std::vector<torch::jit::IValue> Preprocess(
        const torch::Device& device,
        std::map<uint8_t, std::string>& idx_to_req_i,
        std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) = 0;
      
      virtual torch::Tensor Predict(
        std::shared_ptr<torch::jit::script::Module> model, 
        torch::Tensor& inputs,
        const torch::Device& device,
        std::map<uint8_t, std::string>& idx_to_req_i,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) = 0;

      virtual void Postprocess(
        const torch::Tensor& data,
        std::map<uint8_t, std::string>& idx_to_req_i,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) = 0;

      /**
       * @brief 
       * function Predict <=> entry point function handle
       * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L205
       * @param inference_request 
       * @return std::shared_ptr<torchserve::InferenceResponse> 
       */
      void Handle(
        std::shared_ptr<torch::jit::script::Module>& model,
        const torch::Device& device,
        std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
        std::map<uint8_t, std::string> idx_to_req_id;
        auto inputs = Preprocess(device, idx_to_req_id, request_batch, response_batch);
        auto outputs = Predict(model, inputs, device, idx_to_req_id, response_batch);
        Postprocess(outputs, idx_to_req_id, response_batch);
      }

      protected:
      torch::Device GetTorchDevice(
        std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

      std::shared_ptr<torchserve::Manifest> manifest_;
      std::string model_path_;
    };
  } // namespace torchscripted
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_
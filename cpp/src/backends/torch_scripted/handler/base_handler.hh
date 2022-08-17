#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_BASE_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_BASE_HANDLER_HH_

#include <torch/script.h>
#include <torch/torch.h>
#include <functional>
#include <map>
#include <memory>
#include <utility>

#include "src/utils/message.hh"
#include "src/utils/model_archive.hh"

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
      BaseHandler() {};
      virtual ~BaseHandler() {};

      virtual void Initialize(
        const std::string& model_dir,
        std::shared_ptr<torchserve::Manifest>& manifest) {
        model_dir_ = model_dir;
        manifest_ = manifest;
      };

      virtual std::pair<std::shared_ptr<torch::jit::script::Module>, std::shared_ptr<torch::Device>>
      LoadModel(
        std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

      virtual std::vector<torch::jit::IValue> Preprocess(
        std::shared_ptr<torch::Device>& device,
        std::map<uint8_t, std::string>& idx_to_req_id,
        std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);
      
      virtual torch::Tensor Inference(
        std::shared_ptr<torch::jit::script::Module> model, 
        std::vector<torch::jit::IValue>& inputs,
        std::shared_ptr<torch::Device>& device,
        std::map<uint8_t, std::string>& idx_to_req_id,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);

      virtual void Postprocess(
        const torch::Tensor& data,
        std::map<uint8_t, std::string>& idx_to_req_id,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch);

      /**
       * @brief 
       * function Predict <=> entry point function handle
       * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L205
       * @param inference_request 
       * @return std::shared_ptr<torchserve::InferenceResponse> 
       */
      void Handle(
        std::shared_ptr<torch::jit::script::Module>& model,
        std::shared_ptr<torch::Device>& device,
        std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
        try {
          std::map<uint8_t, std::string> idx_to_req_id;
          auto inputs = Preprocess(device, idx_to_req_id, request_batch, response_batch);
          auto outputs = Inference(model, inputs, device, idx_to_req_id, response_batch);
          Postprocess(outputs, idx_to_req_id, response_batch);
        } catch (...) {
          LOG(ERROR) << "Failed to handle this batch";
        }
      }

      protected:
      std::shared_ptr<torch::Device> GetTorchDevice(
        std::shared_ptr<torchserve::LoadModelRequest>& load_model_request);

      std::shared_ptr<torchserve::Manifest> manifest_;
      std::string model_dir_;
    };
  } // namespace torchscripted
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_
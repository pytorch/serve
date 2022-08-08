#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_BASE_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_BASE_HANDLER_HH_

#include <torch/script.h>
#include <torch/torch.h>
#include <functional>
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
        const std::shared_ptr<torchserve::Manifest>& manifest) {
        model_path_ = model_path;
        manifest_ = manifest;
      };

      virtual std::pair<std::shared_ptr<torch::jit::script::Module>, torch::Device> 
      LoadModel(
        std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

      virtual std::vector<torch::jit::IValue> Preprocess(
        torchserve::InferenceRequestBatch batch) = 0;
      
      virtual std::shared_ptr<torchserve::InferenceResponse> Postprocess(
        torch::Tensor data) = 0;

      virtual torch::Tensor Predict(
        std::shared_ptr<torch::jit::script::Module> model,
        std::vector<torch::jit::IValue> inputs) = 0;

      /**
       * @brief 
       * function Predict <=> entry point function handle
       * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L205
       * @param inference_request 
       * @return std::shared_ptr<torchserve::InferenceResponse> 
       */
      std::shared_ptr<torchserve::InferenceResponse> Handle(
        std::shared_ptr<torch::jit::script::Module> model,
        torch::Device device,
        torchserve::InferenceRequestBatch batch) {
        auto inputs = Preprocess(inference_request_batch);
        auto output = Predict(model, inputs);
        return Postprocess(output);
      }

      protected:
      torch::Device GetTorchDevice(
        std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

      std::shared_ptr<torchserve::Manifest> manifest_;
      std::string model_path_;
    };
  } // namespace torchscripted
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_
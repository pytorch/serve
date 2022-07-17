#ifndef TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_

#include <memory>

#include "src/backends/torch_scripted/torch_scripted_backend.hh"

namespace torchserve {
  /**
   * @brief 
   * TorchBaseHandler <=> BaseHandler:
   * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L37
   * 
   * TorchBaseHandler is not responsible for loading model since it is derived from
   * TorchScritpedModelInstance.
   */
  class TorchBaseHandler : public TorchScritpedModelInstance {
    public:
    TorchBaseHandler(
      std::shared_ptr<torch::jit::script::Module> model, 
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request,
      std::shared_ptr<torchserve::Manifest> manifest) : 
      TorchScritpedModelInstance(model, load_model_request, manifest) {};
    virtual ~TorchBaseHandler() {};

    /**
     * @brief 
     * function Predict <=> entry point function handle
     * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L205
     * @param inference_request 
     * @return std::shared_ptr<torchserve::InferenceResponse> 
     */
    std::shared_ptr<torchserve::InferenceResponse> Predict(
      std::unique_ptr<torchserve::InferenceRequest> inference_request) {
      return std::make_shared<torchserve::InferenceResponse>();
    };
  };
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_
#include <fmt/format.h>
#include <memory>

#include "src/backends/torch_scripted/torch_scripted_backend.hh"

namespace torchserve {
  std::pair<
  std::unique_ptr<torchserve::LoadModelResponse>, 
  std::shared_ptr<torchserve::ModelInstance>> 
  TorchScriptedBackend::LoadModelInternal(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request,
    std::shared_ptr<torchserve::Manifest> manifest) {
    std::shared_ptr<torch::jit::script::Module> module;
    try {
      if (load_model_request->gpu_id != -1) {
        module = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(
            fmt::format("{}/{}", 
            load_model_request->model_path, 
            manifest->GetModel().serialized_file),
            GetTorchDevice(load_model_request)));
      } else {
        module = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(fmt::format("{}/{}", 
            load_model_request->model_path, 
            manifest->GetModel().serialized_file)));
      }
    } catch (const c10::Error& e) {
      LOG(ERROR) << "loading the model: " 
      << load_model_request->model_path 
      << ", error: " << e.msg();
      return std::make_pair(
        std::make_unique<torchserve::LoadModelResponse>(
          // TODO: check existing 
          500,
          2,
          e.msg()),
         nullptr);
    }
    /**
     * @brief 
     * TODO: 
     * - load handler shared lib defined in manifest
     * - create model_instance object from the handler
     */
    auto model_instance = std::make_shared<torchserve::TorchScritpedModelInstance>(
      module, load_model_request, manifest);
    return std::make_pair(
      std::make_unique<torchserve::LoadModelResponse>(
        // TODO: check current response msg content
        200,
        2,
        "OK"),
      model_instance);
  }

  torch::Device TorchScriptedBackend::GetTorchDevice(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
    /**
     * @brief 
     * TODO: extend LoadModelRequest to support 
     * - device type: CPU, GPU or others
     */
    if (load_model_request->gpu_id < 0) {
      return torch::kCPU;
    } 

    return torch::Device(torch::kCUDA, load_model_request->gpu_id);
  }

  std::shared_ptr<torchserve::InferenceResponse> TorchScritpedModelInstance::Predict(
    std::unique_ptr<torchserve::InferenceRequest> inference_request) {
    return handler_->Handle(
      std::move(inference_request), 
      [model = model_](std::vector<torch::jit::IValue> inputs) -> torch::Tensor {
        return model->forward(inputs).toTensor();
      });
  }
} //namespace torchserve

extern "C" {
  torchserve::Backend* createBackend() {
    return new torchserve::TorchScriptedBackend();
  }

  void deleteBackend(torchserve::Backend* p) {
    delete p;
  }
}
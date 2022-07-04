#include <fmt/format.h>

#include "src/backends/torch_scripted/torch_scripted_backend.hh"


namespace torchserve {
  std::pair<
  std::unique_ptr<torchserve::LoadModelResponse>, 
  std::shared_ptr<torchserve::ModelInstance>> 
  TorchScriptedBackend::LoadModelInternal(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
    std::shared_ptr<torch::jit::script::Module> module;
    try {
      if (load_model_request->gpu_id != -1) {
        module = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(
            load_model_request->model_path, 
            GetTorchDevice(load_model_request)));
      } else {
        module = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(load_model_request->model_path));
      }
    } catch (const c10::Error& e) {
      LOG(ERROR) << "error loading the model";
      return std::make_pair(
        std::make_unique<torchserve::LoadModelResponse>(
          // TODO: check existing 
          500,
          2,
         "OK"),
         nullptr);
    }
    auto model_instance = std::make_shared<torchserve::TorchScritpedModelInstance>();
    model_instance->Initialize(module, load_model_request);
    return std::make_pair(
      std::make_unique<torchserve::LoadModelResponse>(
        // TODO: check existing 
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
  
  void TorchScritpedModelInstance::Initialize(
    std::shared_ptr<torch::jit::script::Module> model, 
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
    load_model_request_ = load_model_request;
    model_ = model;
    // TODO: set instance_id_ after LoadModelRequest is extended to support 
    // device type: CPU, GPU or others
  }


  // This is the entry point function of libtorch_scripted_backend_xxx.so
  std::shared_ptr<torchserve::Backend> CreateBackend() {
    return std::make_shared<TorchScriptedBackend>();
  }
} //namespace
#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace torchserve {
  namespace torchscripted {
    std::shared_ptr<torch::jit::script::Module> BaseHandler::LoadModel(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
      std::shared_ptr<torch::jit::script::Module> module;
      try {
        module = std::make_shared<torch::jit::script::Module>(torch::jit::load(
          // TODO: windows
          fmt::format("{}/{}", 
          load_model_request->model_path, 
          manifest->GetModel().serialized_file),
          GetTorchDevice(load_model_request)));
        /*
        if (load_model_request->gpu_id != -1) {
          module = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(
              // TODO: windows
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
        */
      } catch (const c10::Error& e) {
        LOG(ERROR) << "loading the model: " 
        << load_model_request->model_name 
        << ", device id:" 
        << load_model_request->gpu_id
        << ", error: " << e.msg();
        throw e;
      }
    }

    std::vector<torch::jit::IValue> BaseHandler::Preprocess(
      const torchserve::InferenceRequestBatch& inference_request_batch) {
      std::vector<std::vector<std::byte>&> batch;
      for(const auto& inference_request : inference_request_batch) {
        batch.emplace(inference_request->)
      }
    }

    torch::Device BaseHandler::GetTorchDevice(
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
  } // namespace torchscripted
} // namespace torchserve

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
  torchserve::torchscripted::BaseHandler *allocatorBaseHandler() {
    return new torchserve::torchscripted::BaseHandler();
  }
  
  void deleterBaseHandler(torchserve::torchscripted::BaseHandler *p) {
    delete p;
  }
}
#endif
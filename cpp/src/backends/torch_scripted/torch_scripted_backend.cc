#include "src/backends/torch_scripted/torch_scripted_backend.hh"

namespace torchserve {
  namespace torchscripted {
    bool Backend::Initialize(const std::string& model_path) {
      if (!torchserve::Backend::Initialize(model_path)) {
        return false;
      }
      LoadHandler();
      if (handler_ == nullptr) {
        return false;
      }
      handler_->Initialize(model_path, manifest_);

      // TODO: support request envelope: 
      // https://github.com/pytorch/serve/tree/master/ts/torch_handler/request_envelope
      return true;
    }

    void Backend::LoadHandler() {
      const std::string& handler_str = manifest_->GetModel().handler;
      std::size_t delimiter_pos = handler_str.find(manifest_->kHandler_Delimiter);
      if (delimiter_pos != std::string::npos) {
        std::string lib_path = handler_str.substr(0, delimiter_pos);
        std::string handler_class_name = handler_str.substr(delimiter_pos + 1);
        std::string allocator_func = fmt::format("allocator{}", handler_class_name);
        std::string deleter_func = fmt::format("deleter{}", handler_class_name);
        dl_loader_ = std::make_unique<torchserve::DLLoader<BaseHandler>>(
          lib_path, allocator_func, deleter_func);
        dl_loader_->OpenDL();
        handler_ = dl_loader_->GetInstance();
      } else {
        handler_ = HandlerFactory::GetInstance().createHandler(handler_str);
      }
    }

    std::pair<
    std::unique_ptr<torchserve::LoadModelResponse>, 
    std::shared_ptr<torchserve::ModelInstance>> 
    Backend::LoadModelInternal(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
      try {
        auto result = handler_->LoadModel(load_model_request);
        auto model_instance = std::make_shared<ModelInstance>(
          BuildModelInstanceId(load_model_request), result.first, handler_, result.second), ;
        return std::make_pair(
          std::make_unique<torchserve::LoadModelResponse>(
            // TODO: check current response msg content
            200,
            2,
            "OK"),
            model_instance);
      } catch (const c10::Error& e) {
        return std::make_pair(
          std::make_unique<torchserve::LoadModelResponse>(
            // TODO: check existing 
            500,
            2,
            e.msg()),
          nullptr);
      }
    }

    std::shared_ptr<torchserve::InferenceResponse> ModelInstance::Predict(
      torchserve::InferenceRequestBatch batch) {
      return handler_->Handle(model_, inference_request_batch);
    }
  } // namespace torchscripted
} //namespace torchserve
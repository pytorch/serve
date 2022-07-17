#include "src/backends/core/backend.hh"

namespace torchserve {
  std::pair<std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<torchserve::ModelInstance>> 
  Backend::LoadModel(std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
    /**
     * TODO: 
     * in multi-thread, this function is called by workers.
     * - check manifest is created 
     * - check the model instance status in LoadModel
     * - status_NOT_INIT: call LoadModelInternal and register the new model instance
     * - status_INIT: wait for notification
     * - status_READY: return the model instance if it is already.
     * 
     * Common steps:
     * https://github.com/pytorch/serve/blob/master/ts/model_loader.py#L62
     */
    manifest_ = std::make_shared<torchserve::Manifest>();
    manifest_->Initialize(fmt::format("{}/MAR-INF/MANIFEST.json", load_model_request->model_path));
    
    return LoadModelInternal(std::move(load_model_request), manifest_);
  }
}

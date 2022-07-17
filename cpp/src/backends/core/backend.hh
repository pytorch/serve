#ifndef TS_CPP_BACKENDS_CORE_BACKEND_HH_
#define TS_CPP_BACKENDS_CORE_BACKEND_HH_

#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <utility>

#include "src/utils/config.hh"
#include "src/utils/message.hh"
#include "src/utils/model_archive.hh"

namespace torchserve {
  /**
   * @brief TorchServe ModelInstance Interface
   * ModelInstance <=> Service:
   *  https://github.com/pytorch/serve/blob/master/ts/service.py#L21
   */
  class ModelInstance {
    public:
    ModelInstance(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request,
      std::shared_ptr<torchserve::Manifest> manifest) :
      load_model_request_(load_model_request), manifest_(manifest) {
      // TODO: set instance_id_ after LoadModelRequest is extended to support 
      // device type: CPU, GPU or others
    };
    virtual ~ModelInstance() {};

    virtual std::shared_ptr<torchserve::InferenceResponse> Predict(
      std::unique_ptr<torchserve::InferenceRequest> inference_request) = 0;

    std::shared_ptr<torchserve::Manifest> GetManifest() {
      return manifest_;
    };

    std::shared_ptr<torchserve::LoadModelRequest> GetLoadModelRequest() {
      return load_model_request_;
    };

    const std::string& GetInstanceId() {
      return instance_id_;
    };

    protected:
    // instance_id naming convention:
    // device_type + ":" + device_id (or object id)
    std::string instance_id_;
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request_;
    std::shared_ptr<torchserve::Manifest> manifest_;
  };

  /**
   * @brief TorchServe Backend Interface
   * Backend <=> ModelLoader:
   * https://github.com/pytorch/serve/blob/master/ts/model_loader.py#L28
   * 
   * Note:
   * Any framework should implement its own backend which includes:
   * 1. Implement class Backend
   * 2. Implement class ModelInstance
   * 3. function std::shared_ptr<Backend> CreateBackend()
   * 
   * The core idea:
   * - A framework has its own backend in a model server.
   * - A backend has multiple model instances.
   * - A worker is associated with one model instance.
   * - A model instance is one model loaded on CPU or GPU.
   */
  class Backend {
    public:
    enum ModelInstanceStatus {
      NOT_INIT,
      INIT,
      READY
    };

    Backend() {};
    ~Backend() {};

    std::pair<std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<ModelInstance>> 
    LoadModel(std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

    virtual 
    std::pair<std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<ModelInstance>> 
    LoadModelInternal(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request,
      std::shared_ptr<torchserve::Manifest> manifest) = 0;

    protected:
    std::shared_ptr<torchserve::Manifest> manifest_;
    // key: model_instance_id
    // value: model_instance_status
    std::map<std::string, torchserve::Backend::ModelInstanceStatus> model_instance_status_;

    // key: model_instance_id
    // value: model_instance    
    std::map<std::string, std::shared_ptr<torchserve::ModelInstance>> model_instance_table_;
  };

  class ModelWorker {
    public:
    ModelWorker() {};
    ~ModelWorker();

    private:
    std::shared_ptr<torchserve::ModelInstance> model_instance_;
  };
}  // namespace torchserve
#endif // TS_CPP_BACKENDS_CORE_BACKEND_HH_
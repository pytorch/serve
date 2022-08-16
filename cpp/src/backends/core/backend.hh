#ifndef TS_CPP_BACKENDS_CORE_BACKEND_HH_
#define TS_CPP_BACKENDS_CORE_BACKEND_HH_

#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "src/utils/config.hh"
#include "src/utils/message.hh"

namespace torchserve {
  /**
   * @brief TorchServe ModelInstance Interface
   * ModelInstance <=> Service:
   *  https://github.com/pytorch/serve/blob/master/ts/service.py#L21
   */
  class ModelInstance {
    public:
    ModelInstance(const std::string& instance_id) : instance_id_(instance_id) {};
    virtual ~ModelInstance() {};

    virtual std::shared_ptr<torchserve::InferenceResponseBatch> Predict(
      std::shared_ptr<torchserve::InferenceRequestBatch> batch) = 0;

    const std::string& GetInstanceId() {
      return instance_id_;
    };

    protected:
    // instance_id naming convention:
    // device_type + ":" + device_id (or object id)
    std::string instance_id_;
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
    virtual ~Backend() {};

    virtual bool Initialize(const std::string& model_path) {
      manifest_ = std::make_shared<torchserve::Manifest>();
      // TODO: windows
      return manifest_->Initialize(fmt::format("{}/MAR-INF/MANIFEST.json", model_path));
    };

    std::pair<std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<ModelInstance>> 
    LoadModel(std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

    virtual 
    std::pair<std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<ModelInstance>> 
    LoadModelInternal(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request) = 0;

    protected:
    std::string BuildModelInstanceId(std::shared_ptr<torchserve::LoadModelRequest> load_model_request);


    std::shared_ptr<torchserve::Manifest> manifest_;
    // key: model_instance_id
    // value: model_instance_status
    std::map<std::string, torchserve::Backend::ModelInstanceStatus> model_instance_status_;

    // key: model_instance_id
    // value: model_instance    
    std::map<std::string, std::shared_ptr<torchserve::ModelInstance>> model_instance_table_;

    std::atomic_uint16_t model_instance_count_ = 0;
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
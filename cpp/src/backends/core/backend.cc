#include "src/backends/core/backend.hh"

#include <memory>

#include "src/backends/handler/handler_factory.hh"
#include "src/utils/logging.hh"

namespace torchserve {
Backend::Backend() {}

Backend::~Backend() {
  handler_.reset();
  model_instance_table_.clear();
  // Todo: do proper cleanup
  // dl_loader_->CloseDL();
}

bool Backend::Initialize(const std::string &model_dir) {
  random_generator_.seed(time(0));
  manifest_ = std::make_shared<torchserve::Manifest>();
  auto manifest_file = fmt::format("{}/MAR-INF/MANIFEST.json", model_dir);
  // TODO: windows
  TS_LOGF(DEBUG, "Initializing from manifest: {}", manifest_file);
  if (!manifest_->Initialize(manifest_file)) {
    TS_LOGF(ERROR, "Could not initialize from manifest: {}", manifest_file);
    return false;
  }

  LoadHandler(model_dir);

  if (!handler_) {
    TS_LOG(ERROR, "Could not load handler");
    return false;
  }

  handler_->Initialize(model_dir, manifest_);

  return true;
}

void Backend::LoadHandler(const std::string &model_dir) {
  const std::string &handler_str = manifest_->GetModel().handler;
  std::size_t delimiter_pos = handler_str.find(manifest_->kHandler_Delimiter);
  if (delimiter_pos != std::string::npos) {
    TS_LOGF(DEBUG, "Loading custom handler: {}", handler_str);
#ifdef __APPLE__
    std::string lib_path = fmt::format("{}/{}.dylib", model_dir,
                                       handler_str.substr(0, delimiter_pos));
#else
    std::string lib_path = fmt::format("{}/{}.so", model_dir,
                                       handler_str.substr(0, delimiter_pos));
#endif
    std::string handler_class_name = handler_str.substr(delimiter_pos + 1);
    std::string allocator_func = fmt::format("allocator{}", handler_class_name);
    std::string deleter_func = fmt::format("deleter{}", handler_class_name);

    dl_loader_ = std::make_unique<DLLoader<BaseHandler>>(
        lib_path, allocator_func, deleter_func);
    dl_loader_->OpenDL();
    handler_ = dl_loader_->GetInstance();
  } else {
    TS_LOGF(DEBUG, "Creating handler: {}", handler_str);
    handler_ = HandlerFactory::GetInstance().createHandler(handler_str);
  }
}

std::unique_ptr<torchserve::LoadModelResponse> Backend::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
  /**
   * TODO:
   * in multi-thread, this function is called by workers.
   * - check the model instance status in LoadModel
   * - status_NOT_INIT: call LoadModelInternal and register the new model
   * instance
   * - status_INIT: wait for notification
   * - status_READY: return the model instance if it is already.
   *
   * Common steps:
   * serve/blob/master/ts/model_loader.py#L62
   */

  // TODO: support request envelope:
  // serve/tree/master/ts/torch_handler/request_envelope

  return LoadModelInternal(std::move(load_model_request));
}

std::unique_ptr<LoadModelResponse> Backend::LoadModelInternal(
    std::shared_ptr<LoadModelRequest> load_model_request) {
  std::string model_instance_id = BuildModelInstanceId(load_model_request);
  try {
    model_instance_table_[model_instance_id] = {
        ModelInstanceStatus::INIT, std::shared_ptr<ModelInstance>(nullptr)};

    auto result = handler_->LoadModel(load_model_request);
    SetModelInstanceInfo(model_instance_id, ModelInstanceStatus::READY,
                         std::make_shared<ModelInstance>(
                             model_instance_id, std::move(result.first),
                             handler_, std::move(result.second)));

    ready_model_instance_ids_.emplace_back(model_instance_id);
    std::string message =
        fmt::format("loaded model {}", load_model_request->model_name);
    return std::make_unique<LoadModelResponse>(
        // TODO: check current response msg content
        200, message);
  } catch (const c10::Error &e) {
    TS_LOGF(ERROR, "Error during model loading: {}", e.what());
    SetModelInstanceInfo(model_instance_id, ModelInstanceStatus::FAILED,
                         std::shared_ptr<ModelInstance>(nullptr));
    return std::make_unique<LoadModelResponse>(
        // TODO: check existing
        500, e.msg());
  }
}

std::string Backend::BuildModelInstanceId(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
  std::string device_type("cpu");
  if (load_model_request->gpu_id >= 0) {
    device_type = "gpu";
  }
  return fmt::format("{}:{}:{}", device_type, load_model_request->gpu_id,
                     model_instance_count_.fetch_add(1));
}

void Backend::SetModelInstanceInfo(
    const std::string &model_instance_id, ModelInstanceStatus new_status,
    std::shared_ptr<torchserve::ModelInstance> new_model_instance) {
  model_instance_table_[model_instance_id].status = new_status;
  model_instance_table_[model_instance_id].model_instance =
      std::move(new_model_instance);
}

torchserve::Backend::ModelInstanceStatus Backend::GetModelInstanceStatus(
    const std::string &model_instance_id) {
  auto model_instance_info = model_instance_table_.find(model_instance_id);
  if (model_instance_info == model_instance_table_.end()) {
    return torchserve::Backend::ModelInstanceStatus::NOT_INIT;
  }
  return model_instance_info->second.status;
}

std::shared_ptr<torchserve::ModelInstance> Backend::GetModelInstance(
    const std::string &model_instance_id) {
  auto model_instance_info = model_instance_table_.find(model_instance_id);
  if (model_instance_info == model_instance_table_.end()) {
    return std::shared_ptr<torchserve::ModelInstance>(nullptr);
  }
  return model_instance_info->second.model_instance;
}

std::shared_ptr<torchserve::ModelInstance> Backend::GetModelInstance() {
  if (ready_model_instance_ids_.empty()) {
    return std::shared_ptr<torchserve::ModelInstance>(nullptr);
  }

  auto model_instance_info =
      model_instance_table_.find(ready_model_instance_ids_[Random()]);
  return model_instance_info->second.model_instance;
}

std::size_t Backend::Random() {
  auto size = ready_model_instance_ids_.size();
  if (size == 1) {
    return 0;
  } else {
    std::uniform_int_distribution<unsigned int> uint_distribution_(0, size - 1);
    return uint_distribution_(random_generator_);
  }
}
}  // namespace torchserve

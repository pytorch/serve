#include <folly/json.h>
#include <glog/logging.h>

#include "src/utils/file_system.hh"
#include "src/utils/model_archive.hh"

namespace torchserve {
  std::shared_ptr<torchserve::Manifest> Manifest::LoadManifest(
    const std::string& manifest_json_file) {
    auto manifest_stream = std::move(torchserve::FileSystem::GetStream(manifest_json_file));
    std::string str((std::istreambuf_iterator<char>(*manifest_stream)), 
    std::istreambuf_iterator<char>(*manifest_stream));
    std::shared_ptr<torchserve::Manifest> ret = std::shared_ptr<torchserve::Manifest>();
    try {
      auto val = folly::parseJson(str);
      ret->create_on = std::move(val[torchserve::Manifest::kCreateOn].asString());
      ret->description = std::move(val[torchserve::Manifest::kDescription].asString());
      ret->archiver_version = std::move(val[torchserve::Manifest::kArchiverVersion].asString());
      ret->runtime_type = std::move(val[torchserve::Manifest::kRuntimeType].asString());
      auto model = val[torchserve::Manifest::kModel];
      ret->model.model_name = std::move(model[torchserve::Manifest::kModelName].asString());
      ret->model.model_version = std::move(model[torchserve::Manifest::kModelVersion].asString());
      ret->model.workflow_name = std::move(model[torchserve::Manifest::kWorkflowName].asString());
      ret->model.description = std::move(model[torchserve::Manifest::kDescription].asString());
      ret->model.handler = std::move(model[torchserve::Manifest::kHandler].asString());
      ret->model.serialized_file = std::move(model[torchserve::Manifest::kSerializedFile].asString());
      ret->model.model_file = std::move(model[torchserve::Manifest::kModelFile].asString());
      ret->model.extensions = std::move(model[torchserve::Manifest::kExtensions].asString());
      ret->model.requirements_file = std::move(model[torchserve::Manifest::kReqirementsFile].asString());
      ret->model.spec_file = std::move(model[torchserve::Manifest::kSpecFile].asString());
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to parse: " << manifest_json_file << ", error: " << e.what();
      throw e;
    }
    return ret;
  }
} // //namespace torchserve
#include <folly/json.h>
#include <glog/logging.h>

#include "src/utils/file_system.hh"
#include "src/utils/model_archive.hh"

namespace torchserve {
  bool Manifest::Initialize(
    const std::string& manifest_json_file_path) {
    try {
      auto manifest_stream = torchserve::FileSystem::GetStream(manifest_json_file_path);
      std::string str((std::istreambuf_iterator<char>(*manifest_stream)), std::istreambuf_iterator<char>());

      auto val = folly::parseJson(str);
      auto model = val[torchserve::Manifest::kModel];
      if (model == NULL) {
        LOG(ERROR) << "Item: model is not defined in " << manifest_json_file_path;
      }

      SetValue(model, torchserve::Manifest::kModel_Handler, model_.handler, true);
      if (!SetValue(
        model, torchserve::Manifest::kModel_SerializedFile, 
        model_.serialized_file, false) && 
        !SetValue(model, torchserve::Manifest::kModel_ModelFile, 
        model_.model_file, false)) {
          LOG(ERROR) << "Item: " << torchserve::Manifest::kModel_SerializedFile 
          << " and item : " << torchserve::Manifest::kModel_ModelFile 
          << " not defined in " << manifest_json_file_path;
      }

      SetValue(model, torchserve::Manifest::kModel_ModelName, model_.model_name, false);
      SetValue(model, torchserve::Manifest::kModel_ModelVersion, model_.model_version, false);
      SetValue(model, torchserve::Manifest::kModel_WorkflowName, model_.workflow_name, false);
      SetValue(model, torchserve::Manifest::kModel_Description, model_.description, false);
      
      SetValue(model, torchserve::Manifest::kModel_Extensions, model_.extensions, false);
      SetValue(model, torchserve::Manifest::kModel_ReqirementsFile, model_.requirements_file, false);
      SetValue(model, torchserve::Manifest::kModel_SpecFile, model_.spec_file, false);

      SetValue(val, torchserve::Manifest::kCreateOn, create_on_, false);
      SetValue(val, torchserve::Manifest::kArchiverVersion, archiver_version_, false);
      SetValue(val, torchserve::Manifest::kRuntimeType, runtime_type_, false);
    } catch (const std::invalid_argument& e) {
      LOG(ERROR) << "Failed to init Manifest from: " << manifest_json_file_path << ", error: " << e.what();
      return false;
    }
    return true;
  }

  bool Manifest::SetValue(
    const folly::dynamic& source, 
    const std::string& key, 
    std::string& dest, 
    bool required) {
    try {
      dest = source[key].asString();
    } catch (const std::out_of_range& e) {
      if (required) {
        LOG(ERROR) << "Item: " << key << " not defined.";
      } else {
        return false;
      }
    }
    return true;
  }
} // //namespace torchserve
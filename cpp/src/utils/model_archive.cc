#include "src/utils/model_archive.hh"

#include <iostream>

using json = nlohmann::json;

namespace torchserve {
bool Manifest::Initialize(const std::string& manifest_json_file_path) {
  try {
    auto manifest_stream =
        torchserve::FileSystem::GetStream(manifest_json_file_path);
    std::string str((std::istreambuf_iterator<char>(*manifest_stream)),
                    std::istreambuf_iterator<char>());

    auto val = json::parse(str);
    auto model = val[torchserve::Manifest::kModel];
    if (model == NULL) {
      TS_LOGF(ERROR, "Item: model is not defined in {}",
              manifest_json_file_path);
    }

    SetValue(model, torchserve::Manifest::kModel_Handler, model_.handler, true);
    SetValue(model, torchserve::Manifest::kModel_SerializedFile,
             model_.serialized_file, false);
    SetValue(model, torchserve::Manifest::kModel_ModelFile, model_.model_file,
             false);

    SetValue(model, torchserve::Manifest::kModel_ModelName, model_.model_name,
             false);
    SetValue(model, torchserve::Manifest::kModel_ModelVersion,
             model_.model_version, false);
    SetValue(model, torchserve::Manifest::kModel_WorkflowName,
             model_.workflow_name, false);
    SetValue(model, torchserve::Manifest::kModel_Description,
             model_.description, false);

    SetValue(model, torchserve::Manifest::kModel_Extensions, model_.extensions,
             false);
    SetValue(model, torchserve::Manifest::kModel_ReqirementsFile,
             model_.requirements_file, false);
    SetValue(model, torchserve::Manifest::kModel_SpecFile, model_.spec_file,
             false);
    SetValue(model, torchserve::Manifest::kModel_Envelope, model_.envelope,
             false);

    SetValue(val, torchserve::Manifest::kCreateOn, create_on_, false);
    SetValue(val, torchserve::Manifest::kArchiverVersion, archiver_version_,
             false);
    SetValue(val, torchserve::Manifest::kRuntimeType, runtime_type_, false);
    return true;
  } catch (const std::invalid_argument& e) {
    TS_LOGF(ERROR, "Failed to init Manifest from: {}, error: ",
            manifest_json_file_path, e.what());
  } catch (...) {
    TS_LOGF(ERROR, "Failed to init Manifest from: {}", manifest_json_file_path);
  }
  return false;
}

bool Manifest::SetValue(const json& source, const std::string& key,
                        std::string& dest, bool required) {
  if(source.contains(key)) {
    dest = source[key].template get<std::string>();
    return true;
  }else{
    if(required)
      TS_LOGF(ERROR, "Item: {} not defined.", key);
    return false;
  }
}
}  // namespace torchserve

#ifndef TS_CPP_UTILS_MODEL_ARCHIVE_HH_
#define TS_CPP_UTILS_MODEL_ARCHIVE_HH_

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <string>

#include "src/utils/file_system.hh"
#include "src/utils/logging.hh"

namespace torchserve {

class Manifest {
 public:
  // TODO: limit to upper case "LDP", "LSP", ...
  using RuntimeType = std::string;

  inline static const std::string kModel_ModelName = "modelName";
  inline static const std::string kModel_ModelVersion = "modelVersion";
  inline static const std::string kModel_WorkflowName = "workflowName";
  inline static const std::string kModel_Description = "description";
  inline static const std::string kModel_Handler = "handler";
  inline static const std::string kModel_Envelope = "envelope";
  inline static const std::string kModel_SerializedFile = "serializedFile";
  inline static const std::string kModel_ModelFile = "modelFile";
  inline static const std::string kModel_Extensions = "extensions";
  inline static const std::string kModel_ReqirementsFile = "requirementsFile";
  inline static const std::string kModel_SpecFile = "specFile";
  inline static const std::string kCreateOn = "createdOn";
  inline static const std::string kArchiverVersion = "archiverVersion";
  inline static const std::string kRuntimeType = "runtime";
  inline static const std::string kModel = "model";
  inline static const char kHandler_Delimiter = ':';

  // Due to https://github.com/llvm/llvm-project/issues/54668,
  // so ignore bugprone-exception-escape
  // NOLINTBEGIN(bugprone-exception-escape)
  Manifest() = default;
  struct Model {
    std::string model_name;
    std::string model_version;
    // For legacy workflow manifest
    std::string workflow_name;
    std::string description;
    std::string handler;
    std::string envelope;
    std::string serialized_file;
    std::string model_file;
    std::string extensions;
    std::string requirements_file;
    std::string spec_file;
  };
  // NOLINTEND(bugprone-exception-escape)

  bool Initialize(const std::string& manifest_json_file_path);

  const std::string& GetCreatOn() { return create_on_; };

  const std::string& GetArchiverVersion() { return archiver_version_; };

  const std::string& GetRuntimeType() { return runtime_type_; };

  const Model& GetModel() { return model_; };

  void SetHandler(const std::string& handler) { model_.handler = handler; }

 private:
  bool SetValue(const nlohmann::json& source, const std::string& key,
                std::string& dest, bool required);

  std::string create_on_;
  std::string archiver_version_;
  RuntimeType runtime_type_;
  Model model_;
};

class ModelArchive {
 public:
  Manifest manifest;

  static ModelArchive* downloadModel(
      const std::vector<std::string>& allowedUrls, std::string& modelStore,
      std::string& url);

  static ModelArchive* downloadModel(
      const std::vector<std::string>& allowedUrls, std::string& modelStore,
      std::string& url, bool s3SseKmsEnabled);
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_MODEL_ARCHIVE_HH_

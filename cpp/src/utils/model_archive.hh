#ifndef TS_CPP_UTILS_MODEL_ARCHIVE_HH_
#define TS_CPP_UTILS_MODEL_ARCHIVE_HH_

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <map>
#include <memory>
#include <string>

namespace torchserve {
  // TODO: limit to upper case "LDP", "LSP", ...
  using RuntimeType = std::string;
  
  class Manifest {
    public:
    inline static const std::string kModel_ModelName = "modelName";
    inline static const std::string kModel_ModelVersion = "modelVersion";
    inline static const std::string kModel_WorkflowName = "workflowName";
    inline static const std::string kModel_Description = "description";
    inline static const std::string kModel_Handler = "handler";
    inline static const std::string kModel_SerializedFile = "serializedFile";
    inline static const std::string kModel_ModelFile = "modelFile";
    inline static const std::string kModel_Extensions = "extensions";
    inline static const std::string kModel_ReqirementsFile = "requirementsFile";
    inline static const std::string kModel_SpecFile = "specFile";
    inline static const std::string kCreateOn = "createdOn";
    inline static const std::string kArchiverVersion = "archiverVersion";
    inline static const std::string kRuntimeType = "runtime";
    inline static const std::string kModel = "model";

    void Initialize(const std::string& manifest_json_file_path);

    struct Model {
      std::string model_name;
      std::string model_version;
      // For legacy workflow manifest
      std::string workflow_name;  
      std::string description;
      std::string handler;
      std::string serialized_file;
      std::string model_file;
      std::string extensions;
      std::string requirements_file;
      std::string spec_file;
    };

    const std::string& GetCreatOn() {
      return create_on_;
    };
    const std::string& GetArchiverVersion() {
      return archiver_version_;
    };
    const std::string& GetRuntimeType() {
      return runtime_type_;
    };
    const Model& GetModel() {
      return model_;
    };

    private:
    bool SetValue(
      const folly::dynamic& source, 
      const std::string& key, 
      std::string& dest, 
      bool required);

    std::string create_on_;
    std::string archiver_version_;
    RuntimeType runtime_type_;
    Model model_;
  };

  class ModelArchive {
    public:
    Manifest manifest;
    
    static ModelArchive *downloadModel(
      const std::vector<std::string> &allowedUrls, 
      std::string &modelStore,
      std::string &url);

    static ModelArchive *downloadModel(
      const std::vector<std::string> &allowedUrls, 
      std::string &modelStore,
      std::string &url, bool s3SseKmsEnabled);
  };
}  // namespace torchserve
#endif // TS_CPP_UTILS_MODEL_ARCHIVE_HH_
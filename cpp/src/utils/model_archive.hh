#ifndef TS_CPP_UTILS_MODEL_ARCHIVE_HH_
#define TS_CPP_UTILS_MODEL_ARCHIVE_HH_

#include <fmt/format.h>
#include <map>
#include <memory>
#include <string>

namespace torchserve {
  // TODO: limit to upper case "LDP", "LSP", ...
  using RuntimeType = std::string;
  
  class Manifest {
    public:
    inline static const std::string kModelName = "modelName";
    inline static const std::string kModelVersion = "modelVersion";
    inline static const std::string kWorkflowName = "workflowName";
    inline static const std::string kDescription = "description";
    inline static const std::string kHandler = "handler";
    inline static const std::string kSerializedFile = "serializedFile";
    inline static const std::string kModelFile = "modelFile";
    inline static const std::string kExtensions = "extensions";
    inline static const std::string kReqirementsFile = "requirementsFile";
    inline static const std::string kSpecFile = "specFile";
    inline static const std::string kCreateOn = "createOn";
    inline static const std::string kArchiverVersion = "archiverVersion";
    inline static const std::string kRuntimeType = "runtime";
    inline static const std::string kModel = "model";
    
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

    std::string create_on;
    std::string description;
    std::string archiver_version;
    RuntimeType runtime_type;
    Model model;

    static std::shared_ptr<Manifest> LoadManifest(const std::string& manifest_json_file);
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
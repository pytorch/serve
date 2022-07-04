#ifndef TS_CPP_UTILS_MODEL_ARCHIVE_HH_
#define TS_CPP_UTILS_MODEL_ARCHIVE_HH_

#include <fmt/format.h>
#include <map>
#include <string>
#include <stdexcept>

namespace torchserve {
  // TODO: limit to upper case "LDP", "LSP", ...
  using RuntimeType = std::string;
  
  class Manifest {
    struct Model {
      std::string modelName;
      std::string modelVersion;
      // For legacy workflow manifest
      std::string workflowName;  
      std::string description;
      std::string handler;
      std::string serializedFile;
      std::string modelFile;
      std::string extensions;
      std::string requirementsFile;
      std::string specFile;
    };

    std::string createOn;
    std::string description;
    std::string archiverVersion;
    RuntimeType runtimeType;
    Model model;
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
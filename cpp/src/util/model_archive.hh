#ifndef CPP_UTIL_MODEL_ARCHIVE_HH_
#define CPP_UTIL_MODEL_ARCHIVE_HH_

#include <fmt/format.h>
#include <map>
#include <string>
#include <stdexcept>

namespace torchserve {
  enum RuntimeType {
    PYTHON,
    PYTHON2,
    PYTHON3,
    LDP,
    LSP
  };

  const std::map<std::string, RuntimeType> runtime_type_table = {
    {"python", PYTHON},
    {"python2", PYTHON2},
    {"python3", PYTHON3},
    {"libtorch_deploy_process", LDP},
    {"libtorch_scripted_process", LSP}
  };

  RuntimeType GetRuntimeType(const std::string& type_lower_case) {
    std::map<std::string, RuntimeType>::const_iterator it = runtime_type_table.find(type_lower_case);
    
    if (it == runtime_type_table.end()) {
      throw std::invalid_argument(fmt::format("invalid runtime type: {}", type_lower_case));
    }
    return it->second;
  }
  
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
#endif // CPP_UTIL_MODEL_ARCHIVE_HH_
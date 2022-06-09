#pragma once

#include <map>
#include <string>

namespace torchserve {
  enum RuntimeType {
    PYTHON,
    PYTHON2,
    PYTHON3,
    LDP,
    LSP
  };

  const std::map<std::string, RuntimeType> runtimeType_mapping = {
    {"python", PYTHON},
    {"python2", PYTHON2},
    {"python3", PYTHON3},
    {"libtorch_deploy_process", LDP},
    {"libtorch_scripted_process", LSP}
  };

  RuntimeType get_runtime_type_from_string(const std::string &type_lower_case) {
    std::map<std::string, RuntimeType>::const_iterator it = runtimeType_mapping.find(type_lower_case);
    
    if (it == runtimeType_mapping.end()) {
      // logger error
      
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
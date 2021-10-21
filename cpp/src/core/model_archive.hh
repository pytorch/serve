#pragma once

#include <string>

namespace torchserve {
enum RuntimeType {
  PYTHON,
  PYTHON2,
  PYTHON3,
  CPP_TORCH_THREADS,
  CPP_TORCH_IPC_UDS,
  CPP_TORCH_IPC_SHM,
};

struct Manifest {
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
      const std::vector<std::string> &allowedUrls, std::string &modelStore,
      std::string &url);

  static ModelArchive *downloadModel(
      const std::vector<std::string> &allowedUrls, std::string &modelStore,
      std::string &url, bool s3SseKmsEnabled);
};
}  // namespace torchserve
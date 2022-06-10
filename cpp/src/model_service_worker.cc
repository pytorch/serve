#include <torch/script.h>
#include <iostream>
#include <memory>
#include <glog/logging.h>

#include "backend/torch_model_service_worker.hh"


int main(int argc, const char* argv[]) {
  // Init logging
  google::InitGoogleLogging("ts_cpp_backend");
  FLAGS_logtostderr = 1;
  // TODO: Set logging format same as python worker
  LOG(INFO) << "Initializing Libtorch backend worker...";

  // Test libtorch dependency
  /*torch::jit::script::Module module;
  try {
    LOG(INFO) << "Loading model file from " << argv[1];
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    LOG(ERROR) << "Error loading the model\n";
    return -1;
  }

  LOG(INFO) << "Model loaded\n";*/

  std::string s_type("unix");
  std::unique_ptr<torchserve::TorchModelServiceWorker> worker = std::make_unique<torchserve::TorchModelServiceWorker>(
          "unix", "/tmp/.9000", "127.0.0.1", "");
  worker->RunServer();
}
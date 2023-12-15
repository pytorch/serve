#include "src/backends/torch_scripted/torch_scripted_backend.hh"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include "src/utils/message.hh"
#include "src/utils/metrics/registry.hh"

namespace torchserve {
class TorchScriptedBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = std::make_shared<torchserve::torchscripted::Backend>();
  }

  void LoadPredict(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request,
      const std::string& model_dir,
      const std::string& inference_input_file_path,
      const std::string& inference_request_id_prefix,
      int inference_expect_code) {
    MetricsRegistry::Initialize("test/resources/metrics/default_config.yaml",
                                MetricsContext::BACKEND);
    backend_->Initialize(model_dir);
    auto result = backend_->LoadModel(std::move(load_model_request));
    ASSERT_EQ(result->code, 200);

    std::ifstream input(inference_input_file_path,
                        std::ios::in | std::ios::binary);
    std::vector<char> image((std::istreambuf_iterator<char>(input)),
                            (std::istreambuf_iterator<char>()));
    input.close();

    auto inference_request_batch =
        std::make_shared<torchserve::InferenceRequestBatch>();
    for (uint8_t i = 0; i < batch_size_; i++) {
      torchserve::InferenceRequest inference_request;
      inference_request.request_id =
          fmt::format("{}_{}", inference_request_id_prefix, i);
      inference_request
          .headers[torchserve::PayloadType::kHEADER_NAME_DATA_TYPE] =
          torchserve::PayloadType::kDATA_TYPE_BYTES;
      inference_request
          .parameters[torchserve::PayloadType::kPARAMETER_NAME_DATA] = image;

      (*inference_request_batch).emplace_back(inference_request);
    }

    auto inference_response_batch =
        backend_->GetModelInstance()->Predict(inference_request_batch);
    for (const auto& kv : *inference_response_batch) {
      ASSERT_EQ(kv.second->code, inference_expect_code);
    }
  };

  uint8_t batch_size_ = 2;
  std::shared_ptr<torchserve::Backend> backend_;
};

TEST_F(TorchScriptedBackendTest, TestLoadPredictBaseHandler) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/torchscript_model/mnist/mnist_handler",
                        "mnist_scripted_v2", -1, "", "", 1, false),
                    "test/resources/torchscript_model/mnist/base_handler",
                    "test/resources/torchscript_model/mnist/0_png.pt",
                    "mnist_ts", 200);
}

TEST_F(TorchScriptedBackendTest, TestLoadPredictMnistHandler) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/torchscript_model/mnist/mnist_handler",
                        "mnist_scripted_v2", -1, "", "", 1, false),
                    "test/resources/torchscript_model/mnist/mnist_handler",
                    "test/resources/torchscript_model/mnist/0_png.pt",
                    "mnist_ts", 200);
}

TEST_F(TorchScriptedBackendTest, TestLoadPredictBabyLlamaHandler) {
  this->LoadPredict(
      std::make_shared<torchserve::LoadModelRequest>(
          "test/resources/torchscript_model/babyllama/babyllama_handler", "llm",
          -1, "", "", 1, false),
      "test/resources/torchscript_model/babyllama/babyllama_handler",
      "test/resources/torchscript_model/babyllama/prompt.txt", "llm_ts", 200);
}

TEST_F(TorchScriptedBackendTest, TestBackendInitWrongModelDir) {
  auto result = backend_->Initialize("test/resources/torchscript_model/mnist");
  ASSERT_EQ(result, false);
}

TEST_F(TorchScriptedBackendTest, TestBackendInitWrongHandler) {
  auto result = backend_->Initialize(
      "test/resources/torchscript_model/mnist/wrong_handler");
  ASSERT_EQ(result, false);
}

TEST_F(TorchScriptedBackendTest, TestLoadModelFailure) {
  backend_->Initialize("test/resources/torchscript_model/mnist/wrong_model");
  auto result =
      backend_->LoadModel(std::make_shared<torchserve::LoadModelRequest>(
          "test/resources/torchscript_model/mnist/wrong_model",
          "mnist_scripted_v2", -1, "", "", 1, false));
  ASSERT_EQ(result->code, 500);
}

TEST_F(TorchScriptedBackendTest, TestLoadPredictMnistHandlerFailure) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/torchscript_model/mnist/mnist_handler",
                        "mnist_scripted_v2", -1, "", "", 1, false),
                    "test/resources/torchscript_model/mnist/mnist_handler",
                    "test/resources/torchscript_model/mnist/0.png", "mnist_ts",
                    500);
}

}  // namespace torchserve

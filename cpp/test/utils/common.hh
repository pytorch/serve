#pragma once
#include <gtest/gtest.h>

#include "src/backends/core/backend.hh"
#include "src/utils/metrics/registry.hh"

class ModelPredictTest : public ::testing::Test {
 protected:
  void SetUp() override { backend_ = std::make_shared<torchserve::Backend>(); }

  void LoadPredict(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request,
      const std::string& model_dir,
      const std::string& inference_input_file_path,
      const std::string& inference_request_id_prefix,
      int inference_expect_code) {
    torchserve::MetricsRegistry::Initialize(
        "resources/metrics/default_config.yaml",
        torchserve::MetricsContext::BACKEND);
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

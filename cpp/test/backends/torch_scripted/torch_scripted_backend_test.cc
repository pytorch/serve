#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include "src/backends/torch_scripted/torch_scripted_backend.hh"
#include "src/utils/message.hh"

namespace torchserve {
  class TorchScriptedBackendTest : public ::testing::Test {
    protected:
    void SetUp() {
      backend_ = std::make_shared<torchserve::torchscripted::Backend>();
      backend_->Initialize("test/resources/torchscript_model/mnist");

      load_model_request_ = std::make_shared<torchserve::LoadModelRequest>(
        "test/resources/torchscript_model/mnist",
        "mnist_scripted_v2",
        -1,
        "",
        "",
        1,
        false
      );
      
    };
    std::shared_ptr<torchserve::Backend> backend_;
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request_;
  };

  TEST_F(TorchScriptedBackendTest, TestLoadModel) {
    auto result = backend_->LoadModel(std::move(load_model_request_));
    ASSERT_EQ(result.first->code, 200);
  }

  TEST_F(TorchScriptedBackendTest, TestPredict) {
    auto load_model_result = backend_->LoadModel(std::move(load_model_request_));
    ASSERT_EQ(load_model_result.first->code, 200);

    std::ifstream input("test/resources/torchscript_model/mnist/0_png.pt", std::ios::in | std::ios::binary);
    std::vector<char> image(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
    input.close();

    torchserve::InferenceRequest inference_request;
    inference_request.request_id = "mnist_ts_1";
    inference_request.headers[torchserve::PayloadType::kHEADER_NAME_DATA_TYPE] = 
      torchserve::PayloadType::kDATA_TYPE_BYTES;
    inference_request.parameters[torchserve::PayloadType::kPARAMETER_NAME_DATA] = image;

    auto inference_request_batch = std::make_shared<torchserve::InferenceRequestBatch>();
    (*inference_request_batch).emplace_back(inference_request);
    auto inference_response_batch = load_model_result.second->Predict(inference_request_batch);
    for (const auto& kv : *inference_response_batch) {
      ASSERT_EQ(kv.second->code, 200);
      std::cerr << "rt:" << torch::pickle_load(kv.second->msg);
    }
  }
} //namespace
#include <gtest/gtest.h>
#include <memory>

#include "src/backends/torch_scripted/torch_scripted_backend.hh"

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

  TEST_F(TorchScriptedBackendTest, TestLoadModelInternal) {
    auto result = backend_->LoadModel(std::move(load_model_request_));
    ASSERT_EQ(result.first->code, 200);
  }
} //namespace
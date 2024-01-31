#include <fmt/format.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include "src/utils/message.hh"
#include "test/utils/common.hh"

TEST_F(ModelPredictTest, TestLoadPredictBaseHandler) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/examples/mnist/mnist_handler",
                        "mnist_scripted_v2", -1, "", "", 1, false),
                    "test/resources/examples/mnist/base_handler",
                    "test/resources/examples/mnist/0_png.pt", "mnist_ts", 200);
}

TEST_F(ModelPredictTest, TestLoadPredictMnistHandler) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/examples/mnist/mnist_handler",
                        "mnist_scripted_v2", -1, "", "", 1, false),
                    "test/resources/examples/mnist/mnist_handler",
                    "test/resources/examples/mnist/0_png.pt", "mnist_ts", 200);
}

TEST_F(ModelPredictTest, TestBackendInitWrongModelDir) {
  auto result = backend_->Initialize("test/resources/examples/mnist");
  ASSERT_EQ(result, false);
}

TEST_F(ModelPredictTest, TestBackendInitWrongHandler) {
  auto result =
      backend_->Initialize("test/resources/examples/mnist/wrong_handler");
  ASSERT_EQ(result, false);
}

TEST_F(ModelPredictTest, TestLoadModelFailure) {
  backend_->Initialize("test/resources/examples/mnist/wrong_model");
  auto result =
      backend_->LoadModel(std::make_shared<torchserve::LoadModelRequest>(
          "test/resources/examples/mnist/wrong_model", "mnist_scripted_v2", -1,
          "", "", 1, false));
  ASSERT_EQ(result->code, 500);
}

TEST_F(ModelPredictTest, TestLoadPredictMnistHandlerFailure) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/examples/mnist/mnist_handler",
                        "mnist_scripted_v2", -1, "", "", 1, false),
                    "test/resources/examples/mnist/mnist_handler",
                    "test/resources/examples/mnist/0.png", "mnist_ts", 500);
}

#include <gtest/gtest.h>

#include "protocol/mock_socket.hh"
#include "src/backends/core/backend.hh"
#include "src/backends/process/model_worker.hh"
#include "src/utils/metrics/registry.hh"

namespace torchserve {
TEST(BackendIntegTest, TestOTFProtocolAndHandler) {
  auto client_socket = std::make_shared<MockSocket>();
  // mock socket for load cmd
  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 1);
        strncpy(data, "L", length);
      }));
  auto cmd = OTFMessage::RetrieveCmd(*client_socket);
  ASSERT_EQ(cmd, 'L');

  // mock socket for load model request
  EXPECT_CALL(*client_socket, RetrieveInt())
      .Times(6)
      // model_name length
      .WillOnce(::testing::Return(5))
      // model_path length
      .WillOnce(::testing::Return(37))
      // batch_size
      .WillOnce(::testing::Return(1))
      // handler length
      .WillOnce(::testing::Return(11))
      // gpu_id
      .WillOnce(::testing::Return(-1))
      // envelope length
      .WillOnce(::testing::Return(0));
  EXPECT_CALL(*client_socket, RetrieveBool())
      .Times(1)
      .WillOnce(::testing::Return(true));
  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(4)
      // model_name
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 5);
        strncpy(data, "mnist", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 37);
        strncpy(data, "resources/examples/mnist/base_handler", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 11);
        strncpy(data, "BaseHandler", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 0);
        strncpy(data, "", length);
      }));

  EXPECT_CALL(*client_socket, SendAll(testing::_, testing::_)).Times(1);
  auto load_model_request = OTFMessage::RetrieveLoadMsg(*client_socket);
  ASSERT_EQ(load_model_request->model_dir,
            "resources/examples/mnist/base_handler");
  ASSERT_EQ(load_model_request->model_name, "mnist");
  ASSERT_EQ(load_model_request->envelope, "");
  ASSERT_EQ(load_model_request->model_name, "mnist");
  ASSERT_EQ(load_model_request->batch_size, 1);
  ASSERT_EQ(load_model_request->gpu_id, -1);

  // initialize backend
  auto backend = std::make_shared<torchserve::Backend>();
  MetricsRegistry::Initialize("resources/metrics/default_config.yaml",
                              MetricsContext::BACKEND);
  backend->Initialize("resources/examples/mnist/base_handler");

  // load the model
  auto load_model_response = backend->LoadModel(load_model_request);
  ASSERT_EQ(load_model_response->code, 200);

  // send load model response to socket
  torchserve::OTFMessage::SendLoadModelResponse(*client_socket,
                                                std::move(load_model_response));

  // mock socket for inference cmd
  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 1);
        strncpy(data, "I", length);
      }));
  cmd = OTFMessage::RetrieveCmd(*client_socket);
  ASSERT_EQ(cmd, 'I');

  // mock socket for inference request
  EXPECT_CALL(*client_socket, RetrieveInt())
      .Times(7)
      // request_id length
      .WillOnce(::testing::Return(4))
      // end of headers
      .WillOnce(::testing::Return(-1))
      // parameter_name length
      .WillOnce(::testing::Return(4))
      // content_type length
      .WillOnce(::testing::Return(4))
      // value length
      .WillOnce(::testing::Return(3883))
      // end of parameters
      .WillOnce(::testing::Return(-1))
      // end of request
      .WillOnce(::testing::Return(-1));

  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(4)
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "reqi", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "body", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "cont", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 3883);
        // strncpy(data, "valu", length);
        std::ifstream input("resources/examples/mnist/0_png.pt",
                            std::ios::in | std::ios::binary);
        std::vector<char> image((std::istreambuf_iterator<char>(input)),
                                (std::istreambuf_iterator<char>()));
        memcpy(data, image.data(), 3883);
        input.close();
      }));

  EXPECT_CALL(*client_socket, SendAll(testing::_, testing::_)).Times(1);
  auto batch_inference_request =
      OTFMessage::RetrieveInferenceMsg(*client_socket);
  auto inference_request = batch_inference_request->at(0);
  ASSERT_EQ(batch_inference_request->size(), 1);
  ASSERT_EQ(inference_request.headers.size(), 2);
  ASSERT_EQ(inference_request.parameters.size(), 1);
  ASSERT_EQ(inference_request.request_id, "reqi");
  ASSERT_EQ(inference_request.headers["body_dtype"], "bytes");
  ASSERT_EQ(inference_request.headers["body:contentType"], "cont");
  ASSERT_EQ(inference_request.parameters["body"].size(), 3883);

  // call handler to run inference
  auto inference_response_batch =
      backend->GetModelInstance()->Predict(batch_inference_request);
  auto prediction =
      torch::pickle_load((*inference_response_batch)["reqi"]->msg).toTensor();
  std::vector<float> expected_result = {0.0000,   -28.5285, -22.8017, -32.5117,
                                        -33.5584, -29.8429, -25.7716, -25.9097,
                                        -27.6592, -24.5729};
  auto tensor_options = torch::TensorOptions().dtype(at::kFloat);
  auto expected_tensor =
      torch::from_blob(expected_result.data(),
                       {static_cast<long long>(expected_result.size())},
                       tensor_options)
          .clone();
  ASSERT_EQ(inference_response_batch->size(), 1);
  ASSERT_EQ((*inference_response_batch)["reqi"]->code, 200);
  ASSERT_EQ(torch::allclose(prediction, expected_tensor), true);

  // send inference response to socket
  torchserve::OTFMessage::SendInferenceResponse(*client_socket,
                                                inference_response_batch);
}
}  // namespace torchserve

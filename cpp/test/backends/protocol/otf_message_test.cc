#include <gtest/gtest.h>

#include "mock_socket.hh"

#include <cstring>

namespace torchserve {
TEST(OTFMessageTest, TestRetieveCmd) {
  auto client_socket = std::make_shared<MockSocket>();
  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 1);
        strncpy(data, "L", length);
      }));
  auto cmd = OTFMessage::RetrieveCmd(*client_socket);
  ASSERT_EQ(cmd, 'L');
}

TEST(OTFMessageTest, TestEncodeLoadModelResponse) {
  std::string message = "model_loaded";
  auto dummyResponse =
      std::make_unique<torchserve::LoadModelResponse>(200, message);
  std::vector<char> data_buffer{};
  torchserve::OTFMessage::EncodeLoadModelResponse(std::move(dummyResponse),
                                                  data_buffer);
  const char* expectedResponse =
      "\x00\x00\x00\xc8\x00\x00\x00\x0cmodel_loaded\xff\xff\xff\xff";
  EXPECT_TRUE(0 == std::memcmp(data_buffer.data(), expectedResponse,
                               data_buffer.size()));
}

TEST(OTFMessageTest, TestUTF8EncodeLoadModelResponse) {
  std::string message = "测试";
  auto dummyResponse =
      std::make_unique<torchserve::LoadModelResponse>(200, message);
  std::vector<char> data_buffer{};
  torchserve::OTFMessage::EncodeLoadModelResponse(std::move(dummyResponse),
                                                  data_buffer);
  const char* expectedResponse =
      "\x00\x00\x00\xc8\x00\x00\x00\x06\xe6\xb5\x8b\xe8\xaf\x95\xff\xff\xff"
      "\xff";
  EXPECT_TRUE(0 == std::memcmp(data_buffer.data(), expectedResponse,
                               data_buffer.size()));
}

TEST(OTFMessageTest, TestRetrieveMsgLoadGpu) {
  std::unique_ptr<MockSocket> client_socket = std::make_unique<MockSocket>();
  LoadModelRequest expected("model_path", "测试", 1, "handler", "envelope", 1,
                            false);
  EXPECT_CALL(*client_socket, RetrieveInt())
      .Times(6)
      // model_name length
      .WillOnce(::testing::Return(6))
      // model_path length
      .WillOnce(::testing::Return(10))
      // batch_size
      .WillOnce(::testing::Return(1))
      // handler length
      .WillOnce(::testing::Return(7))
      // gpu_id
      .WillOnce(::testing::Return(1))
      // envelope length
      .WillOnce(::testing::Return(8));
  EXPECT_CALL(*client_socket, RetrieveBool())
      .Times(1)
      .WillOnce(::testing::Return(false));
  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(4)
      // model_name
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 6);
        strncpy(data, "测试", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 10);
        strncpy(data, "model_path", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 7);
        strncpy(data, "handler", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 8);
        strncpy(data, "envelope", length);
      }));
  auto load_model_request = OTFMessage::RetrieveLoadMsg(*client_socket);
  ASSERT_TRUE(*load_model_request == expected);
  client_socket.reset();
}

TEST(OTFMessageTest, TestRetrieveMsgLoadNoGpu) {
  std::unique_ptr<MockSocket> client_socket = std::make_unique<MockSocket>();
  LoadModelRequest expected("model_path", "model_name", -1, "handler",
                            "envelope", 1, true);
  EXPECT_CALL(*client_socket, RetrieveInt())
      .Times(6)
      .WillOnce(::testing::Return(10))
      .WillOnce(::testing::Return(10))
      .WillOnce(::testing::Return(1))
      .WillOnce(::testing::Return(7))
      .WillOnce(::testing::Return(-1))
      .WillOnce(::testing::Return(8));
  EXPECT_CALL(*client_socket, RetrieveBool())
      .Times(1)
      .WillOnce(::testing::Return(true));
  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(4)
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 10);
        strncpy(data, "model_name", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 10);
        strncpy(data, "model_path", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 7);
        strncpy(data, "handler", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 8);
        strncpy(data, "envelope", length);
      }));
  auto load_model_request = OTFMessage::RetrieveLoadMsg(*client_socket);
  ASSERT_TRUE(*load_model_request == expected);
  client_socket.reset();
}

TEST(OTFMessageTest, TestEncodeSuccessInferenceResponse) {
  std::string request_id = "d22dd8d8-0abf";
  auto inference_response = std::make_shared<InferenceResponse>(request_id);
  inference_response->SetResponse(200, "data_type", "string", "sample_message");
  auto inference_response_batch = std::make_shared<InferenceResponseBatch>();

  (*inference_response_batch)[request_id] = inference_response;
  std::vector<char> data_buffer{};
  OTFMessage::EncodeInferenceResponse(inference_response_batch, data_buffer);
  const char* expectedResponse =
      "\x00\x00\x00\xc8\x00\x00\x00\x12Prediction "
      "success\x00\x00\x00\rd22dd8d8-"
      "0abf\x00\x00\x00\x00\x00\x00\x00\xc8\x00\x00\x00\x00\x00\x00\x00\x01\x00"
      "\x00\x00\tdata_type\x00\x00\x00\x06string\x00\x00\x00\x0esample_"
      "message\xff\xff\xff\xff";
  EXPECT_TRUE(0 == std::memcmp(data_buffer.data(), expectedResponse,
                               data_buffer.size()));
}

TEST(OTFMessageTest, TestEncodeFailureInferenceResponse) {
  std::string request_id = "d22dd8d8-0abf";
  auto inference_response = std::make_shared<InferenceResponse>(request_id);
  inference_response->SetResponse(500, "data_type", "string",
                                  "response_failure_message");
  auto inference_response_batch = std::make_shared<InferenceResponseBatch>();

  (*inference_response_batch)[request_id] = inference_response;
  std::vector<char> data_buffer{};
  OTFMessage::EncodeInferenceResponse(inference_response_batch, data_buffer);
  TS_LOG(ERROR, "result_size: {}", data_buffer.size());
  const char* expectedResponse =
      "\x00\x00\x01\xf4\x00\x00\x00\x18response_failure_"
      "message\x00\x00\x00\rd22dd8d8-"
      "0abf\x00\x00\x00\x00\x00\x00\x01\xf4\x00\x00\x00\x00\x00\x00\x00\x01\x00"
      "\x00\x00\tdata_type\x00\x00\x00\x06string\x00\x00\x00\x18response_"
      "failure_message\xff\xff\xff\xff";
  EXPECT_TRUE(0 == std::memcmp(data_buffer.data(), expectedResponse,
                               data_buffer.size()));
}

TEST(OTFMessageTest, TestRetrieveInferenceMsg) {
  auto client_socket = std::make_shared<MockSocket>();
  EXPECT_CALL(*client_socket, RetrieveInt())
      .Times(9)
      // request_id length
      .WillOnce(::testing::Return(4))
      // header_key length
      .WillOnce(::testing::Return(4))
      // header_value length
      .WillOnce(::testing::Return(4))
      // end of headers
      .WillOnce(::testing::Return(-1))
      // parameter_name length
      .WillOnce(::testing::Return(4))
      // content_type length
      .WillOnce(::testing::Return(4))
      // value length
      .WillOnce(::testing::Return(4))
      // end of parameters
      .WillOnce(::testing::Return(-1))
      // end of request
      .WillOnce(::testing::Return(-1));

  EXPECT_CALL(*client_socket, RetrieveBuffer(testing::_, testing::_))
      .Times(6)
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "reqi", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "heak", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "heav", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "parn", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "cont", length);
      }))
      .WillOnce(testing::Invoke([=](size_t length, char* data) {
        ASSERT_EQ(length, 4);
        strncpy(data, "valu", length);
      }));
  auto batch_inference_request =
      OTFMessage::RetrieveInferenceMsg(*client_socket);
  auto inference_request = batch_inference_request->at(0);
  ASSERT_EQ(batch_inference_request->size(), 1);
  ASSERT_EQ(inference_request.headers.size(), 3);
  ASSERT_EQ(inference_request.parameters.size(), 1);
  ASSERT_EQ(inference_request.request_id, "reqi");
  ASSERT_EQ(inference_request.headers["heak"], "heav");
  ASSERT_EQ(inference_request.headers["body_dtype"], "bytes");
  ASSERT_EQ(inference_request.headers["parn:contentType"], "cont");
  ASSERT_EQ(
      torchserve::Converter::VectorToStr(inference_request.parameters["parn"]),
      "valu");
}
}  // namespace torchserve

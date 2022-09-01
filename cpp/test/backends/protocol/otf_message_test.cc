#include <gtest/gtest.h>

#include "mock_socket.hh"

namespace torchserve {
  TEST(OTFMessageTest, TestEncodeLoadModelResponse) {
    std::string message = "model_loaded";
    auto dummyResponse = std::make_unique<torchserve::LoadModelResponse>(
        200, message.size(), message);
    char *data = new char[sizeof(LoadModelResponse)];
    torchserve::OTFMessage::EncodeLoadModelResponse(std::move(dummyResponse), data);
    const char* expectedResponse = "\x00\x00\x00\xc8\x00\x00\x00\x0cmodel_loaded\xff\xff\xff\xff";
    EXPECT_TRUE(0 == std::memcmp(data, expectedResponse, sizeof(data)));
  }

  TEST(OTFMessageTest, TestUTF8EncodeLoadModelResponse) {
    std::string message = "测试";
    auto dummyResponse = std::make_unique<torchserve::LoadModelResponse>(
        200, message.size(), message);
    char *data = new char[sizeof(LoadModelResponse)];
    torchserve::OTFMessage::EncodeLoadModelResponse(std::move(dummyResponse), data);
    const char* expectedResponse = "\x00\x00\x00\xc8\x00\x00\x00\x06\xe6\xb5\x8b\xe8\xaf\x95\xff\xff\xff\xff";
    EXPECT_TRUE(0 == std::memcmp(data, expectedResponse, sizeof(data)));
  }

  TEST(OTFMessageTest, TestRetrieveMsgLoadGpu) {
    MockSocket *client_socket = new MockSocket();
    LoadModelRequest expected("model_path", "测试", 1, "handler", "envelope", 1, false);
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
                  strcpy(data, "测试");
                }))
                .WillOnce(testing::Invoke([=](size_t length, char* data) {
                  ASSERT_EQ(length, 10);
                  strcpy(data, "model_path");
                }))
                .WillOnce(testing::Invoke([=](size_t length, char* data) {
                  ASSERT_EQ(length, 7);
                  strcpy(data, "handler");
                }))
                .WillOnce(testing::Invoke([=](size_t length, char* data) {
                  ASSERT_EQ(length, 8);
                  strcpy(data, "envelope");
                }));
    auto load_model_request = OTFMessage::RetrieveLoadMsg(*client_socket);
    ASSERT_TRUE(*load_model_request == expected);
    delete client_socket;
  }

  TEST(OTFMessageTest, TestRetrieveMsgLoadNoGpu) {
    MockSocket *client_socket = new MockSocket();
    LoadModelRequest expected("model_path", "model_name", -1, "handler", "envelope", 1, true);
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
                  strcpy(data, "model_name");
                }))
                .WillOnce(testing::Invoke([=](size_t length, char* data) {
                  ASSERT_EQ(length, 10);
                  strcpy(data, "model_path");
                }))
                .WillOnce(testing::Invoke([=](size_t length, char* data) {
                  ASSERT_EQ(length, 7);
                  strcpy(data, "handler");
                }))
                .WillOnce(testing::Invoke([=](size_t length, char* data) {
                  ASSERT_EQ(length, 8);
                  strcpy(data, "envelope");
                }));
    auto load_model_request = OTFMessage::RetrieveLoadMsg(*client_socket);
    ASSERT_TRUE(*load_model_request == expected);
    delete client_socket;
  }
}
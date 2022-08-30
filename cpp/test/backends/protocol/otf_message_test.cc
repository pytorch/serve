#include <gtest/gtest.h>

#include "src/backends/protocol/otf_message.hh"

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
}
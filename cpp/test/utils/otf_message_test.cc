#include <gtest/gtest.h>

#include "src/backends/protocol/otf_message.hh"

namespace torchserve {

  TEST(OTFMessageTest, TestEncodeInferenceResponse) {
    std::string request_id = "d22dd8d8-0abf";
    auto inference_response = std::make_shared<InferenceResponse>(request_id);
    inference_response->SetResponse(
        200,
        "data_type",
        "string",
        "sample_message"
    );
    auto inference_response_batch = std::make_shared<InferenceResponseBatch>();

    (*inference_response_batch)[request_id] = inference_response;
    std::vector<char> data_buffer{};
    OTFMessage::EncodeInferenceResponse(inference_response_batch, data_buffer);
    const char* expectedResponse = "\x00\x00\x00\xc8\x00\x00\x00\x00\x00\x00\x00\rd22dd8d8-0abf\x00\x00\x00\x00\x00\x00\x00\xc8\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\tdata_type\x00\x00\x00\x06string\x00\x00\x00\x0esample_message\xff\xff\xff\xff";
    EXPECT_TRUE(0 == std::memcmp(data_buffer.data(), expectedResponse, data_buffer.size()));
  }
}

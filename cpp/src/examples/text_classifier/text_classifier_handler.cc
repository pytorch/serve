#include "src/examples/text_classifier/text_classifier_handler.hh"

#include <folly/json.h>

namespace text_classifier {

std::vector<torch::jit::IValue> TextClassifierHandler::Preprocess(
    std::shared_ptr<torch::Device> &device,
    std::map<uint8_t, std::string> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch> &request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  auto batch_1 = c10::impl::GenericList(c10::StringType::get());
  auto batch_2 = c10::impl::GenericList(c10::StringType::get());

  uint8_t idx = 0;
  for (auto &request : *request_batch) {
    auto data_it =
        request.parameters.find(torchserve::PayloadType::kDATA_TYPE_STRING);
    auto dtype_it =
        request.headers.find(torchserve::PayloadType::kHEADER_NAME_DATA_TYPE);

    if (data_it == request.parameters.end() ||
        dtype_it == request.headers.end()) {
      TS_LOGF(ERROR, "Empty payload for request id: {}", request.request_id);
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kCONTENT_TYPE_TEXT,
                            "Empty payload");
      continue;
    }

    try {
      std::string json_str(data_it->second.begin(), data_it->second.end());
      auto values = folly::parseJson(json_str);
      batch_1.emplace_back(values["sequence_0"].asString());
      batch_2.emplace_back(values["sequence_1"].asString());
      idx_to_req_id[idx++] = request.request_id;
    } catch (const std::runtime_error &e) {
      TS_LOGF(ERROR, "Failed to load data for request id: {}, error: {}",
              request.request_id, e.what());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
    } catch (const c10::Error &e) {
      TS_LOGF(ERROR, "Failed to load data for request id: {}, c10 error: {}",
              request.request_id, e.msg());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
    }
  }
  return {batch_1, batch_2};
}

void TextClassifierHandler::Postprocess(
    const torch::Tensor &data, std::map<uint8_t, std::string> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  for (const auto &kv : idx_to_req_id) {
    try {
      auto response = (*response_batch)[kv.second];
      response->SetResponse(200, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_BYTES,
                            torch::pickle_save(torch::argmax(data[kv.first])));
    } catch (const std::runtime_error &e) {
      LOG(ERROR) << "Failed to load tensor for request id:" << kv.second
                 << ", error: " << e.what();
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
      throw e;
    } catch (const c10::Error &e) {
      LOG(ERROR) << "Failed to load tensor for request id:" << kv.second
                 << ", c10 error: " << e.msg();
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
      throw e;
    }
  }
}
}  // namespace text_classifier

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::torchscripted::BaseHandler *allocatorTextClassifierHandler() {
  return new text_classifier::TextClassifierHandler();
}

void deleterTextClassifierHandler(torchserve::torchscripted::BaseHandler *p) {
  if (p != nullptr) {
    delete static_cast<text_classifier::TextClassifierHandler *>(p);
  }
}
}
#endif
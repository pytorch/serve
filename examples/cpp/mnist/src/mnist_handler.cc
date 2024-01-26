#include "mnist_handler.hh"

namespace mnist {
void MnistHandler::Postprocess(
    c10::IValue& data,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  auto data_tensor = data.toTensor();
  for (const auto& kv : idx_to_req_id.second) {
    try {
      auto response = (*response_batch)[kv.second];
      response->SetResponse(
          200, "data_tpye", torchserve::PayloadType::kDATA_TYPE_BYTES,
          torch::pickle_save(torch::argmax(data_tensor[kv.first])));
    } catch (const std::runtime_error& e) {
      LOG(ERROR) << "Failed to load tensor for request id:" << kv.second
                 << ", error: " << e.what();
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
      throw e;
    } catch (const c10::Error& e) {
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
}  // namespace mnist

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::BaseHandler* allocatorMnistHandler() {
  return new mnist::MnistHandler();
}

void deleterMnistHandler(torchserve::BaseHandler* p) {
  if (p != nullptr) {
    delete static_cast<mnist::MnistHandler*>(p);
  }
}
}
#endif

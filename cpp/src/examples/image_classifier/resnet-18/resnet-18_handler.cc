#include "src/examples/image_classifier/resnet-18/resnet-18_handler.hh"

namespace resnet {
void ResnetHandler::Postprocess(
    const torch::Tensor& data,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  for (const auto& kv : idx_to_req_id.second) {
    try {
      auto response = (*response_batch)[kv.second];
      namespace F = torch::nn::functional;
      torch::Tensor new_data =
          F::softmax(data[kv.first], F::SoftmaxFuncOptions(1));
      // auto topk_result = at::topk(new_data, 5, 1);
      response->SetResponse(200, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_BYTES,
                            torch::pickle_save(new_data));
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
}  // namespace resnet

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::torchscripted::BaseHandler* allocatorResnetHandler() {
  return new resnet::ResnetHandler();
}

void deleterResnetHandler(torchserve::torchscripted::BaseHandler* p) {
  if (p != nullptr) {
    delete static_cast<resnet::ResnetHandler*>(p);
  }
}
}
#endif

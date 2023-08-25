#include "base_handler.hh"

namespace torchserve {

void BaseHandler::Handle(
    std::shared_ptr<void> model, std::shared_ptr<torch::Device>& device,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  std::string req_ids = "";
  std::map<uint8_t, std::string> map_idx_to_req_id;
  std::pair<std::string&, std::map<uint8_t, std::string>&> idx_to_req_id(
      req_ids, map_idx_to_req_id);
  try {
    auto start_time = std::chrono::system_clock::now();
    auto inputs =
        Preprocess(device, idx_to_req_id, request_batch, response_batch);
    auto outputs =
        Inference(model, inputs, device, idx_to_req_id, response_batch);
    Postprocess(outputs, idx_to_req_id, response_batch);
    auto stop_time = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> duration = stop_time - start_time;
    try {
      auto& handler_time_metric =
          torchserve::MetricsRegistry::GetMetricsCacheInstance()->GetMetric(
              torchserve::MetricType::GAUGE, "HandlerTime");
      handler_time_metric.AddOrUpdate(
          std::vector<std::string>{manifest_->GetModel().model_name, "Model"},
          idx_to_req_id.first, duration.count());
    } catch (std::runtime_error& e) {
      TS_LOG(ERROR, e.what());
    } catch (std::invalid_argument& e) {
      TS_LOGF(ERROR, "Failed to record HandlerTime metric. {}", e.what());
    }
    try {
      auto& prediction_time_metric =
          torchserve::MetricsRegistry::GetMetricsCacheInstance()->GetMetric(
              torchserve::MetricType::GAUGE, "PredictionTime");
      prediction_time_metric.AddOrUpdate(
          std::vector<std::string>{manifest_->GetModel().model_name, "Model"},
          idx_to_req_id.first, duration.count());
    } catch (std::runtime_error& e) {
      TS_LOG(ERROR, e.what());
    } catch (std::invalid_argument& e) {
      TS_LOGF(ERROR, "Failed to record PredictionTime metric. {}", e.what());
    }
  } catch (...) {
    TS_LOG(ERROR, "Failed to handle this batch");
  }
}

std::shared_ptr<torch::Device> BaseHandler::GetTorchDevice(
    std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
  /**
   * @brief
   * TODO: extend LoadModelRequest to support
   * - device type: CPU, GPU or others
   */
  if (load_model_request->gpu_id < 0) {
    return std::make_shared<torch::Device>(torch::kCPU);
  }

  return std::make_shared<torch::Device>(torch::kCUDA,
                                         load_model_request->gpu_id);
}

std::vector<torch::jit::IValue> BaseHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  /**
   * @brief
   * Ref:
   * https://github.com/pytorch/serve/blob/be5ff32dab0d81ceb1c2a9d42550ed5904ae9282/ts/torch_handler/vision_handler.py#L33
   */
  std::vector<torch::jit::IValue> batch_ivalue;
  std::vector<torch::Tensor> batch_tensors;
  uint8_t idx = 0;
  for (auto& request : *request_batch) {
    (*response_batch)[request.request_id] =
        std::make_shared<torchserve::InferenceResponse>(request.request_id);
    idx_to_req_id.first += idx_to_req_id.first.empty()
                               ? request.request_id
                               : "," + request.request_id;
    auto data_it =
        request.parameters.find(torchserve::PayloadType::kPARAMETER_NAME_DATA);
    auto dtype_it =
        request.headers.find(torchserve::PayloadType::kHEADER_NAME_DATA_TYPE);
    if (data_it == request.parameters.end()) {
      data_it = request.parameters.find(
          torchserve::PayloadType::kPARAMETER_NAME_BODY);
      dtype_it =
          request.headers.find(torchserve::PayloadType::kHEADER_NAME_BODY_TYPE);
    }

    if (data_it == request.parameters.end() ||
        dtype_it == request.headers.end()) {
      TS_LOGF(ERROR, "Empty payload for request id: {}", request.request_id);
      (*response_batch)[request.request_id]->SetResponse(
          500, "data_type", torchserve::PayloadType::kCONTENT_TYPE_TEXT,
          "Empty payload");
      continue;
    }
    /*
    case2: the image is sent as string of bytesarray
    if (dtype_it->second == "String") {
      try {
        auto b64decoded_str = folly::base64Decode(data_it->second);
        torchserve::Converter::StrToBytes(b64decoded_str, image);
      } catch (folly::base64_decode_error e) {
        TS_LOGF(ERROR, "Failed to base64Decode for request id: {}, error: {}",
                request.request_id,
                e.what());
      }
    }
    */

    try {
      if (dtype_it->second == torchserve::PayloadType::kDATA_TYPE_BYTES) {
        // case2: the image is sent as bytesarray
        // torch::serialize::InputArchive archive;
        // archive.load_from(std::istringstream
        // iss(std::string(data_it->second)));
        /*
        std::istringstream iss(std::string(data_it->second.begin(),
        data_it->second.end())); torch::serialize::InputArchive archive;
        images.emplace_back(archive.load_from(iss, torch::Device device);

        std::vector<char> bytes(
          static_cast<char>(*data_it->second.begin()),
          static_cast<char>(*data_it->second.end()));

        images.emplace_back(torch::pickle_load(bytes).toTensor().to(*device));
        */
        batch_tensors.emplace_back(
            torch::pickle_load(data_it->second).toTensor().to(*device));
        idx_to_req_id.second[idx++] = request.request_id;
      } else if (dtype_it->second == "List") {
        // case3: the image is a list
      }
    } catch (const std::runtime_error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, error: {}",
              request.request_id, e.what());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
    } catch (const c10::Error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, c10 error: {}",
              request.request_id, e.msg());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
    }
  }
  if (!batch_tensors.empty()) {
    batch_ivalue.emplace_back(torch::stack(batch_tensors).to(*device));
  }

  return batch_ivalue;
}

torch::Tensor BaseHandler::Inference(
    std::shared_ptr<void> model, std::vector<torch::jit::IValue>& inputs,
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  if (device == nullptr) {
    TS_LOG(WARN, "device is nullptr");
  }
  try {
    torch::NoGradGuard no_grad;
    std::shared_ptr<torch::jit::Module> jit_model(
        std::static_pointer_cast<torch::jit::Module>(model));
    return jit_model->forward(inputs).toTensor();
  } catch (const std::runtime_error& e) {
    TS_LOGF(ERROR, "Failed to predict, error: {}", e.what());
    for (auto& kv : idx_to_req_id.second) {
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to inference");
    }
    throw e;
  }
}

void BaseHandler::Postprocess(
    const torch::Tensor& data,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  for (const auto& kv : idx_to_req_id.second) {
    try {
      auto response = (*response_batch)[kv.second];
      response->SetResponse(200, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_BYTES,
                            torch::pickle_save(data[kv.first]));
    } catch (const std::runtime_error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, error: {}",
              kv.second, e.what());
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to postprocess tensor");
    } catch (const c10::Error& e) {
      TS_LOGF(ERROR,
              "Failed to postprocess tensor for request id: {}, error: {}",
              kv.second, e.msg());
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to postprocess tensor");
    }
  }
}
}  // namespace torchserve

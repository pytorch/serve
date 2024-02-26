#include "bert_handler.hh"
#include "src/utils/file_system.hh"

#include <fmt/format.h>
#include <iostream>
#include <torch/csrc/inductor/aoti_model_container_runner.h>
#include <torch/csrc/inductor/aoti_model_container_runner_cuda.h>
#include <typeinfo>

namespace bert {
std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>>
BertCppHandler::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
  try {
    TS_LOG(INFO, "start LoadModel");
    auto device = GetTorchDevice(load_model_request);

    const std::string modelConfigYamlFilePath =
        fmt::format("{}/{}", load_model_request->model_dir, "model-config.yaml");
    model_config_yaml_ = std::make_unique<YAML::Node>(YAML::LoadFile(modelConfigYamlFilePath));

    const std::string mapFilePath =
        fmt::format("{}/{}", load_model_request->model_dir,
        (*model_config_yaml_)["handler"]["mapping"].as<std::string>());
    mapping_json_ = torchserve::FileSystem::LoadJsonFile(mapFilePath);

    max_length_ = (*model_config_yaml_)["handler"]["max_length"].as<int>();

    std::string tokenizer_path =
        fmt::format("{}/{}", load_model_request->model_dir,
        (*model_config_yaml_)["handler"]["tokenizer_path"].as<std::string>());
    auto tokenizer_blob = torchserve::FileSystem::LoadBytesFromFile(tokenizer_path);
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);

    std::string model_so_path =
        fmt::format("{}/{}", load_model_request->model_dir,
        (*model_config_yaml_)["handler"]["model_so_path"].as<std::string>());

    c10::InferenceMode mode;

    if (device->is_cuda()) {
      return std::make_pair(
        std::make_shared<torch::inductor::AOTIModelContainerRunnerCuda>(model_so_path.c_str(), 1, device->str().c_str()),
        device);
    } else {
      return std::make_pair(
        std::make_shared<torch::inductor::AOTIModelContainerRunnerCpu>(model_so_path.c_str()),
        device);
    }
  } catch (const c10::Error& e) {
    TS_LOGF(ERROR, "loading the model: {}, device id: {}, error: {}",
            load_model_request->model_name, load_model_request->gpu_id,
            e.msg());
    throw e;
  } catch (const std::runtime_error& e) {
    TS_LOGF(ERROR, "loading the model: {}, device id: {}, error: {}",
            load_model_request->model_name, load_model_request->gpu_id,
            e.what());
    throw e;
  }
}

c10::IValue BertCppHandler::Preprocess(
    std::shared_ptr<torch::Device> &device,
    std::pair<std::string &, std::map<uint8_t, std::string> &> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch> &request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  auto options = torch::TensorOptions().dtype(torch::kLong);
  auto attention_mask = torch::zeros({static_cast<long>(request_batch->size()), max_length_}, torch::kLong);
  auto batch_tokens = torch::full({static_cast<long>(request_batch->size()), max_length_}, tokenizer_->TokenToId("<pad>"), torch::kLong);

  uint8_t idx = 0;
  for (auto& request : *request_batch) {
    try {
      (*response_batch)[request.request_id] =
          std::make_shared<torchserve::InferenceResponse>(request.request_id);
      idx_to_req_id.first += idx_to_req_id.first.empty()
                                 ? request.request_id
                                 : "," + request.request_id;

      auto data_it = request.parameters.find(
          torchserve::PayloadType::kPARAMETER_NAME_DATA);
      auto dtype_it =
          request.headers.find(torchserve::PayloadType::kHEADER_NAME_DATA_TYPE);
      if (data_it == request.parameters.end()) {
        data_it = request.parameters.find(
            torchserve::PayloadType::kPARAMETER_NAME_BODY);
        dtype_it = request.headers.find(
            torchserve::PayloadType::kHEADER_NAME_BODY_TYPE);
      }

      if (data_it == request.parameters.end() ||
          dtype_it == request.headers.end()) {
        (*response_batch)[request.request_id]->SetResponse(
            500, "data_type", torchserve::PayloadType::kCONTENT_TYPE_TEXT,
            "Empty payload");
        continue;
      }

      std::string msg = torchserve::Converter::VectorToStr(data_it->second);
      // tokenization
      std::vector<int32_t> token_ids = tokenizer_->Encode(msg);;
      int cur_token_ids_length = (int)token_ids.size();
      if (cur_token_ids_length > max_length_) {
        TS_LOGF(ERROR, "prompt too long ({} tokens, max {})", cur_token_ids_length,  max_length_);
      }
      for (int i = 0; i < std::min(cur_token_ids_length, max_length_); i++) {
        attention_mask[idx][i] = 1;
        batch_tokens[idx][i] = token_ids[i];
      }

      idx_to_req_id.second[idx++] = request.request_id;
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
  auto batch_ivalue = c10::impl::GenericList(torch::TensorType::get());
  batch_ivalue.emplace_back(batch_tokens.to(*device));
  batch_ivalue.emplace_back(attention_mask.to(*device));

  return batch_ivalue;
}

c10::IValue BertCppHandler::Inference(
    std::shared_ptr<void> model, c10::IValue &inputs,
    std::shared_ptr<torch::Device> &device,
    std::pair<std::string &, std::map<uint8_t, std::string> &> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  c10::InferenceMode mode;
  try {
    std::shared_ptr<torch::inductor::AOTIModelContainerRunner> runner;
    if (device->is_cuda()) {
      runner = std::static_pointer_cast<torch::inductor::AOTIModelContainerRunnerCuda>(model);
    } else {
      runner = std::static_pointer_cast<torch::inductor::AOTIModelContainerRunnerCpu>(model);
    }

    auto batch_output_tensor_vector = runner->run(inputs.toTensorVector());
    return c10::IValue(batch_output_tensor_vector[0]);
  } catch (std::runtime_error& e) {
    TS_LOG(ERROR, e.what());
  } catch (const c10::Error& e) {
    TS_LOGF(ERROR, "Failed to apply inference on input, c10 error:{}", e.msg());
  }
}

void BertCppHandler::Postprocess(
    c10::IValue &inputs,
    std::pair<std::string &, std::map<uint8_t, std::string> &> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  auto& data = inputs.toTensor();
  for (const auto &kv : idx_to_req_id.second) {
    try {
      auto out = data[kv.first].unsqueeze(0);
      auto y_hat = torch::argmax(out, 1).item<int>();
      auto predicted_idx = std::to_string(y_hat);
      auto response = (*response_batch)[kv.second];

      response->SetResponse(200, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            torchserve::FileSystem::GetJsonValue(mapping_json_, predicted_idx).asString());
    } catch (const std::runtime_error &e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, error: {}",
              kv.second, e.what());
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to postprocess tensor");
    } catch (const c10::Error &e) {
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
}  // namespace bert

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::BaseHandler *allocatorBertCppHandler() {
  return new bert::BertCppHandler();
}

void deleterBertCppHandler(torchserve::BaseHandler *p) {
  if (p != nullptr) {
    delete static_cast<bert::BertCppHandler *>(p);
  }
}
}
#endif

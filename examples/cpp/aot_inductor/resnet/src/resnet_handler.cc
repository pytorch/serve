#include "resnet_handler.hh"

#include <iostream>
#include <typeinfo>
#include <unordered_map>

#include <fmt/format.h>
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 2
  #include <torch/csrc/inductor/aoti_model_container_runner.h>
  #include <torch/csrc/inductor/aoti_model_container_runner_cuda.h>
#else
  #include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
  #include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

#include "src/utils/file_system.hh"

namespace resnet {
std::string ResnetCppHandler::MapClassToLabel(const torch::Tensor& classes, const torch::Tensor& probs) {
  std::unordered_map<std::string, float> map;
  for (int i = 0; i < classes.sizes()[0]; i++) {
    auto class_value = mapping_json_->GetValue(std::to_string(classes[i].item<long>()));
    map[class_value.GetValue(1).AsString()] = probs[i].item<float>();
  }
  std::string json_string = "{\n";
  for(auto p : map) {
    json_string += fmt::format("{}:{},\n", p.first, p.second);
  }
  json_string.pop_back();
  json_string.pop_back();

  return json_string + "\n}";
}

std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>>
ResnetCppHandler::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
  try {
    auto device = GetTorchDevice(load_model_request);

    const std::string modelConfigYamlFilePath =
        fmt::format("{}/{}", load_model_request->model_dir, "model-config.yaml");
    model_config_yaml_ = std::make_unique<YAML::Node>(YAML::LoadFile(modelConfigYamlFilePath));

    const std::string mapFilePath =
      fmt::format("{}/{}", load_model_request->model_dir,
        (*model_config_yaml_)["handler"]["mapping"].as<std::string>());
    mapping_json_ = std::make_unique<torchserve::Json>(torchserve::Json::ParseJsonFile(mapFilePath));

    std::string model_so_path =
      fmt::format("{}/{}", load_model_request->model_dir,
        (*model_config_yaml_)["handler"]["model_so_path"].as<std::string>());
    c10::InferenceMode mode;

    #if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 2    
      if (device->is_cuda()) {
        return std::make_pair(
          std::make_shared<torch::inductor::AOTIModelContainerRunnerCuda>(model_so_path.c_str()),
          device);
      } else {
        return std::make_pair(
          std::make_shared<torch::inductor::AOTIModelContainerRunnerCpu>(model_so_path.c_str()),
          device);
      }
    #else
      if (device->is_cuda()) {
        return std::make_pair(
          std::make_shared<torch::inductor::AOTIModelContainerRunnerCuda>(model_so_path, 1, device->str()),
          device);
      } else {
        return std::make_pair(
          std::make_shared<torch::inductor::AOTIModelContainerRunnerCpu>(model_so_path),
          device);
      }
    #endif
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

c10::IValue ResnetCppHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  auto batch_ivalue = c10::impl::GenericList(c10::TensorType::get());

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

    try {
      if (dtype_it->second == torchserve::PayloadType::kDATA_TYPE_BYTES) {
        batch_tensors.emplace_back(
            torch::pickle_load(data_it->second).toTensor());
        idx_to_req_id.second[idx++] = request.request_id;
      } else {
        TS_LOG(ERROR, "Not supported input format, only support bytesstring in this example");
        (*response_batch)[request.request_id]->SetResponse(
          500, "data_type", torchserve::PayloadType::kCONTENT_TYPE_TEXT,
          "Not supported input format, only support bytesstring in this example");
        continue;
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

c10::IValue ResnetCppHandler::Inference(
    std::shared_ptr<void> model, c10::IValue &inputs,
    std::shared_ptr<torch::Device> &device,
    std::pair<std::string &, std::map<uint8_t, std::string> &> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  c10::InferenceMode mode;
  auto batch_ivalue = c10::impl::GenericList(c10::TensorType::get());
  try {
    std::shared_ptr<torch::inductor::AOTIModelContainerRunner> runner;
    if (device->is_cuda()) {
      runner = std::static_pointer_cast<torch::inductor::AOTIModelContainerRunnerCuda>(model);
    } else {
      runner = std::static_pointer_cast<torch::inductor::AOTIModelContainerRunnerCpu>(model);
    }
    auto data = inputs.toTensorList()[0].get().toTensor();
    std::vector<torch::Tensor> input_vec;
    input_vec.emplace_back(data);
    auto batch_output_tensor_vector = runner->run(input_vec);
    batch_ivalue.emplace_back(torch::stack(batch_output_tensor_vector).to(*device));
  } catch (std::runtime_error& e) {
    TS_LOG(ERROR, e.what());
  } catch (const c10::Error& e) {
    TS_LOGF(ERROR, "Failed to apply inference on input, c10 error:{}", e.msg());
  }
  return batch_ivalue;
}

void ResnetCppHandler::Postprocess(
    c10::IValue &inputs,
    std::pair<std::string &, std::map<uint8_t, std::string> &> &idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch> &response_batch) {
  auto data = inputs.toTensorList().get(0);
  auto ps = torch::softmax(data[0], 1);
  auto top5 = torch::topk(ps, 5, 1);
  for (const auto &kv : idx_to_req_id.second) {
    try {
      auto probs = std::get<0>(top5)[kv.first];
      auto classes = std::get<1>(top5)[kv.first];
      auto response = (*response_batch)[kv.second];
      response->SetResponse(200, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            MapClassToLabel(classes, probs));
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
}  // namespace resnet

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::BaseHandler *allocatorResnetCppHandler() {
  return new resnet::ResnetCppHandler();
}

void deleterResnetCppHandler(torchserve::BaseHandler *p) {
  if (p != nullptr) {
    delete static_cast<resnet::ResnetCppHandler *>(p);
  }
}
}
#endif

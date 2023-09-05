#include "src/examples/babyllama/baby_llama_handler.hh"

#include <typeinfo>

namespace llm {

std::pair<std::shared_ptr<torch::jit::script::Module>,
          std::shared_ptr<torch::Device>>
LlmHandler::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
  try {
    auto device = GetTorchDevice(load_model_request);
    // Load dummy model
    auto module = std::make_shared<torch::jit::script::Module>(
        torch::jit::load(fmt::format("{}/{}", load_model_request->model_dir,
                                     manifest_->GetModel().serialized_file),
                         *device));

    const std::string configFilePath =
        fmt::format("{}/{}", load_model_request->model_dir, "config.json");
    std::string jsonContent;
    if (!folly::readFile(configFilePath.c_str(), jsonContent)) {
      std::cerr << "config.json not found at: " << configFilePath << std::endl;
      throw;
    }
    folly::dynamic json;
    json = folly::parseJson(jsonContent);
    std::string checkpoint_path;
    std::string tokenizer_path;
    if (json.find("checkpoint_path") != json.items().end() &&
        json.find("tokenizer_path") != json.items().end()) {
      checkpoint_path = json["checkpoint_path"].asString();
      tokenizer_path = json["tokenizer_path"].asString();
    } else {
      std::cerr
          << "Required fields 'model_name' and 'model_path' not found in JSON."
          << std::endl;
      throw;
    }

    build_transformer(&transformer, const_cast<char*>(checkpoint_path.c_str()));

    build_tokenizer(&tokenizer, const_cast<char*>(tokenizer_path.c_str()),
                    transformer.config.vocab_size);

    float temperature =
        1.0f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well,
                        // but slower
    unsigned long long rng_seed;
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp,
                  rng_seed);

    return std::make_pair(module, device);
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

std::vector<torch::jit::IValue> LlmHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  std::vector<torch::jit::IValue> batch_ivalue;
  std::vector<torch::Tensor> batch_tensors;
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
        TS_LOGF(ERROR, "Empty payload for request id: {}", request.request_id);
        (*response_batch)[request.request_id]->SetResponse(
            500, "data_type", torchserve::PayloadType::kCONTENT_TYPE_TEXT,
            "Empty payload");
        continue;
      }

      std::string msg = torchserve::Converter::VectorToStr(data_it->second);

      int num_prompt_tokens = 0;

      std::unique_ptr<char[], void (*)(char*)> msgCStr(
          new char[msg.size() + 1], [](char* ptr) { delete[] ptr; });

      std::strcpy(msgCStr.get(), msg.c_str());

      std::unique_ptr<int[]> prompt_tokens(new int[msg.length()]);

      encode(&tokenizer, msgCStr.get(), 1, 0, prompt_tokens.get(),
             &num_prompt_tokens);

      std::vector<torch::Tensor> tensor_vector;
      for (int64_t i = 0; i < num_prompt_tokens; ++i) {
        int token = prompt_tokens[i];
        torch::Tensor tensor = torch::tensor(token, torch::kInt64);
        tensor_vector.push_back(tensor);
      }
      torch::Tensor stacked_tensor = torch::stack(tensor_vector);
      batch_ivalue.push_back(stacked_tensor);

      idx_to_req_id.second[idx++] = request.request_id;

    } catch (const std::runtime_error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, error: {}",
              request.request_id, e.what());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
    } catch (const c10::Error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, c10 error:{}",
              request.request_id, e.msg());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
    }
  }

  return batch_ivalue;
}

torch::Tensor LlmHandler::Inference(
    std::shared_ptr<torch::jit::script::Module> model,
    std::vector<torch::jit::IValue>& inputs,
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  std::vector<torch::Tensor> tensor_vector;
  tensor_vector.reserve(steps);
  torch::Tensor tokens_list_tensor = inputs[0].toTensor();

  int64_t num_elements = tokens_list_tensor.numel();

  // // Convert the tensor to a vector of long values
  std::vector<long> long_vector;
  long_vector.reserve(num_elements);

  int64_t* data_ptr = tokens_list_tensor.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_elements; ++i) {
    long_vector.push_back(data_ptr[i]);
  }

  std::unique_ptr<int[]> prompt_tokens(new int[num_elements]);
  for (int64_t i = 0; i < num_elements; ++i) {
    prompt_tokens[i] = static_cast<int>(long_vector[i]);
  }

  // start the main loop
  long start =
      0;     // used to time our code, only initialized after first iteration
  int next;  // will store the next token in the sequence
  int token = prompt_tokens[0];  // kick off with the first token in the prompt
  int pos = 0;                   // position in the sequence
  while (pos < steps) {
    // forward the transformer to get logits for the next token
    float* logits = forward(&transformer, token, pos);

    // advance the state state machine
    if (pos < num_elements - 1) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(&sampler, logits);
    }
    pos++;

    torch::Tensor tensor = torch::tensor(next, torch::kLong);
    tensor_vector.push_back(tensor);

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }

  // report achieved tok/s (pos-1 because the timer starts after first
  // iteration)
  if (pos > 1) {
    long end = time_in_ms();
    double token_per_sec = (pos - 1) / (double)(end - start) * 1000;
    std::cout << "Achieved tok per sec: " << token_per_sec << std::endl;
  }

  torch::Tensor stacked_tensor = torch::stack(tensor_vector);
  return stacked_tensor;
}

void LlmHandler::Postprocess(
    const torch::Tensor& data,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  for (const auto& kv : idx_to_req_id.second) {
    try {
      int64_t num_elements = data.numel();
      int64_t* data_ptr = data.data_ptr<int64_t>();
      int64_t token = 1;
      std::string concatenated_string;
      for (int64_t i = 0; i < num_elements; ++i) {
        char* piece = decode(&tokenizer, token, data_ptr[i]);
        std::string piece_string(piece);
        token = data_ptr[i];
        concatenated_string += piece_string;
      }

      std::cout << "Generated String:  " << concatenated_string << std::endl;

      auto response = (*response_batch)[kv.second];

      response->SetResponse(200, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            concatenated_string);
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

}  // namespace llm

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::torchscripted::BaseHandler* allocatorLlmHandler() {
  return new llm::LlmHandler();
}

void deleterLlmHandler(torchserve::torchscripted::BaseHandler* p) {
  if (p != nullptr) {
    delete static_cast<llm::LlmHandler*>(p);
  }
}
}
#endif
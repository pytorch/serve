#include "src/examples/llamacpp/llamacpp_handler.hh"

#include <torch/script.h>
#include <torch/torch.h>

#include <typeinfo>

namespace llm {

void LlamaCppHandler::initialize_context() {
  llama_ctx = llama_new_context_with_model(llamamodel, ctx_params);

  if (llama_ctx == nullptr) {
    std::cerr << "Failed to initialize llama context" << std::endl;
  } else {
    std::cout << "Context initialized successfully" << std::endl;
  }
}

std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>>
LlamaCppHandler::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
  try {
    auto device = GetTorchDevice(load_model_request);

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
    if (json.find("checkpoint_path") != json.items().end()) {
      checkpoint_path = json["checkpoint_path"].asString();
    } else {
      std::cerr << "Required field 'checkpoint_path' not found in JSON."
                << std::endl;
      throw;
    }
    params.model = checkpoint_path;
    params.main_gpu = 0;
    params.n_gpu_layers = 35;

    llama_backend_init(params.numa);
    ctx_params = llama_context_default_params();
    model_params = llama_model_default_params();
    llamamodel = llama_load_model_from_file(params.model.c_str(), model_params);

    return std::make_pair(nullptr, device);
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

c10::IValue LlamaCppHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  initialize_context();

  auto batch_ivalue = c10::impl::GenericList(torch::TensorType::get());
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

      // tokenization

      std::vector<llama_token> tokens_list;
      tokens_list = ::llama_tokenize(llama_ctx, msg, true);

      // const int max_context_size = llama_n_ctx(ctx);
      const int max_tokens_list_size = max_context_size - 4;

      if ((int)tokens_list.size() > max_tokens_list_size) {
        std::cout << __func__ << ": error: prompt too long ("
                  << tokens_list.size() << " tokens, max "
                  << max_tokens_list_size << ")\n";
      }

      // Print the tokens from the prompt :
      std::vector<torch::Tensor> tensor_vector;
      for (auto id : tokens_list) {
        torch::Tensor tensor = torch::tensor(id, torch::kInt64);
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
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, c10 error: {}",
              request.request_id, e.msg());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
    }
  }

  return batch_ivalue;
}

c10::IValue LlamaCppHandler::Inference(
    std::shared_ptr<void> model, c10::IValue& inputs,
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  torch::InferenceMode guard;
  std::vector<torch::Tensor> batch_output_vector;
  for (const auto input : inputs.toTensorList()) {
    torch::Tensor tokens_list_tensor = input.get().toTensor();

    int64_t num_elements = tokens_list_tensor.numel();

    int64_t* data_ptr = tokens_list_tensor.data_ptr<int64_t>();
    std::vector<llama_token> tokens_list;

    for (int64_t i = 0; i < num_elements; ++i) {
      tokens_list.push_back(data_ptr[i]);
    }
    const int n_gen = std::min(32, max_context_size);

    long pos = 0;
    while (pos < n_gen) {
      // evaluate the transformer

      if (llama_eval(llama_ctx, tokens_list.data(), int(tokens_list.size()),
                     llama_get_kv_cache_token_count(llama_ctx))) {
        std::cout << "Failed to eval\n" << __func__ << std::endl;
        break;
      }

      tokens_list.clear();

      // sample the next token

      llama_token new_token_id = 0;

      auto logits = llama_get_logits(llama_ctx);
      auto n_vocab = llama_n_vocab(llamamodel);

      std::vector<llama_token_data> candidates;
      candidates.reserve(n_vocab);

      for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(
            llama_token_data{token_id, logits[token_id], 0.0f});
      }

      llama_token_data_array candidates_p = {candidates.data(),
                                             candidates.size(), false};

      new_token_id = llama_sample_token_greedy(llama_ctx, &candidates_p);

      // is it an end of stream ?
      if (new_token_id == llama_token_eos(llamamodel)) {
        std::cout << "Reached [end of text]\n";
        break;
      }

      // print the new token :
      std::cout << "New Token: "
                << llama_token_to_piece(llama_ctx, new_token_id) << std::endl;

      // push this new token for next evaluation
      tokens_list.push_back(new_token_id);
      pos += 1;
    }

    std::vector<torch::Tensor> tensor_vector;
    for (auto id : tokens_list) {
      torch::Tensor tensor = torch::tensor(id, torch::kLong);
      tensor_vector.push_back(tensor);
    }

    torch::Tensor stacked_tensor = torch::stack(tensor_vector);
    batch_output_vector.push_back(stacked_tensor);
  }

  llama_print_timings(llama_ctx);
  return torch::stack(batch_output_vector);
}

void LlamaCppHandler::Postprocess(
    c10::IValue& output,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  for (const auto& kv : idx_to_req_id.second) {
    auto data = output.toTensorList();
    try {
      int64_t num_elements = data[kv.first].get().toTensor().numel();

      // Convert the tensor to a vector of long values
      std::stringstream generated_text_stream;

      auto data_ptr = data[kv.first].get().toTensor().data_ptr<int64_t>();
      for (int64_t i = 0; i < num_elements; ++i) {
        generated_text_stream << llama_token_to_piece(llama_ctx, data_ptr[i]);
      }

      std::string generated_text_str = generated_text_stream.str();
      std::cout << "Generated Text Str: " << generated_text_str << std::endl;

      auto response = (*response_batch)[kv.second];

      response->SetResponse(200, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            generated_text_str);
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

LlamaCppHandler::~LlamaCppHandler() noexcept {
  llama_free(llama_ctx);
  llama_free_model(llamamodel);
  llama_backend_free();
}

}  // namespace llm

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::BaseHandler* allocatorLlamaCppHandler() {
  return new llm::LlamaCppHandler();
}

void deleterLlamaCppHandler(torchserve::BaseHandler* p) {
  if (p != nullptr) {
    delete static_cast<llm::LlamaCppHandler*>(p);
  }
}
}
#endif

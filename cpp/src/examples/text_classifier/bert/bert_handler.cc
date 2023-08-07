#include "src/examples/image_classifier/bert/bert_handler.hh"

#include <typeinfo>

#include "src/examples/image_classifier/bert/tokenizer/libtorch_demo/tokenizers_binding/remote_rust_tokenizer.h"

namespace bert {

std::vector<torch::jit::IValue> BertHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  std::vector<torch::jit::IValue> batch_ivalue;
  std::vector<torch::Tensor> batch_tensors;
  uint8_t idx = 0;
  for (auto& request : *request_batch) {
    std::vector new_data = request.parameters["data"];
    std::string msg = torchserve::Converter::VectorToStr(new_data);

    const size_t seq_len = 128;
    const long start_token = 101;
    const long end_token = 102;
    std::vector<long> input_ids(seq_len, 0);
    input_ids[0] = start_token;
    size_t pos = 1;                // current write position in input_ids
    std::string delimiter = "\n";  // Change this to the appropriate
    size_t delimiter_pos = msg.find(delimiter);
    std::string sentence_1 = msg.substr(0, delimiter_pos);
    std::string sentence_2 = msg.substr(delimiter_pos + delimiter.length());

    // tokenize sentence_1 and copy to output buffer
    std::vector<uint32_t> buffer(seq_len, 0);
    remote_rust_encode(sentence_1.c_str(), buffer.data(), buffer.size());
    for (size_t i = 0; i < seq_len && buffer[i]; i++, pos++) {
      input_ids[pos] = buffer[i];
    }

    // mark end of sentence_1
    input_ids[pos++] = end_token;
    const size_t sentence_2_start = pos;

    std::fill(buffer.begin(), buffer.end(), 0);
    remote_rust_encode(sentence_2.c_str(), buffer.data(), buffer.size());
    for (size_t i = 0; i < seq_len && buffer[i]; i++, pos++) {
      input_ids[pos] = buffer[i];
    }

    // mark end of sentence_2
    input_ids[pos++] = end_token;

    // construct attention mask
    std::vector<long> attention_mask(seq_len, 0);
    for (size_t i = 0; i < seq_len; ++i)
      attention_mask[i] = input_ids[i] ? 1 : 0;

    // token type ids are 0s for sentence_1 (incl. separators), 1s for
    // sentence_2
    std::vector<long> token_type_ids(seq_len, 0);
    for (size_t i = sentence_2_start; i < seq_len; i++) {
      if (!attention_mask[i]) break;
      token_type_ids[i] = 1;
    }

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
        torch::Tensor input_ids_tensor =
            torch::from_blob(input_ids.data(),
                             {static_cast<long>(input_ids.size())},
                             torch::kLong)
                .clone();
        torch::Tensor attention_mask_tensor =
            torch::from_blob(attention_mask.data(),
                             {static_cast<long>(attention_mask.size())},
                             torch::kLong)
                .clone();
        torch::Tensor token_type_ids_tensor =
            torch::from_blob(token_type_ids.data(),
                             {static_cast<long>(token_type_ids.size())},
                             torch::kLong)
                .clone();

        batch_ivalue.push_back(torch::jit::IValue(input_ids_tensor));
        batch_ivalue.push_back(torch::jit::IValue(attention_mask_tensor));
        batch_ivalue.push_back(torch::jit::IValue(token_type_ids_tensor));
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

  return batch_ivalue;
}

torch::Tensor BertHandler::Inference(
    std::shared_ptr<torch::jit::script::Module> model,
    std::vector<torch::jit::IValue>& inputs,
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> first_three_elements(inputs.begin(),
                                                       inputs.begin() + 3);

  std::vector<torch::jit::IValue> modelinputs;
  modelinputs.push_back(
      torch::jit::IValue(first_three_elements[0]).toTensor().unsqueeze(0));
  modelinputs.push_back(
      torch::jit::IValue(first_three_elements[1]).toTensor().unsqueeze(0));
  modelinputs.push_back(
      torch::jit::IValue(first_three_elements[2]).toTensor().unsqueeze(0));

  auto outputs = model->forward(modelinputs);
  auto tensor_output = outputs.toTuple()->elements()[0].toTensor();

  return tensor_output;
}

void BertHandler::Postprocess(
    const torch::Tensor& data,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {

  for (const auto& kv : idx_to_req_id.second) {
    try {
      auto max_result = torch::argmax(data);
      bool isParaphrase = (max_result.item<int64_t>() == 1);
      auto response = (*response_batch)[kv.second];
      if (isParaphrase) {
        std::cout << "Paraphrase";
        response->SetResponse(200, "data_type",
                              torchserve::PayloadType::kDATA_TYPE_BYTES,
                              "paraphrase");
      } else {
        std::cout << "Not paraphrase";
        response->SetResponse(200, "data_type",
                              torchserve::PayloadType::kDATA_TYPE_BYTES,
                              "not paraphrase");
      }
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
}  // namespace bert

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::torchscripted::BaseHandler* allocatorBertHandler() {
  return new bert::BertHandler();
}

void deleterBertHandler(torchserve::torchscripted::BaseHandler* p) {
  if (p != nullptr) {
    delete static_cast<bert::BertHandler*>(p);
  }
}
}
#endif

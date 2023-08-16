#include "src/examples/image_classifier/llm/llm_handler.hh"

#include <torch/script.h>
#include <torch/torch.h>

#include <typeinfo>

#include "examples/common.h"
#include "ggml.h"
#include "llama.h"

namespace llm {

std::vector<torch::jit::IValue> LlmHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  /**
   * @brief
   * Ref:
   * https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py#L27
   */

  // Model Loading
  gpt_params params;
  params.model = "/home/ubuntu/serve/cpp/llama-2-7b-chat.ggmlv3.q4_0.bin";
  params.prompt = "Hello my name is";
  llama_backend_init(params.numa);


  llama_model* model;
  llama_context* ctx;

  std::tie(model, ctx) = llama_init_from_gpt_params(params);

  if (model == NULL) {
    std::cout << "<<<<<<<<<<<<Unable to load the model" << std::endl;
  }

  // Tokenization

  std::vector<llama_token> tokens_list;
  tokens_list = ::llama_tokenize(ctx, params.prompt, true);

  // const int max_context_size = llama_n_ctx(ctx);
  const int max_context_size = 64;
  const int max_tokens_list_size = max_context_size - 4;

  if ((int)tokens_list.size() > max_tokens_list_size) {
    std::cout << __func__ << ": error: prompt too long (" << tokens_list.size()
              << " tokens, max " << max_tokens_list_size << ")\n";
  }

  // Print the tokens from the prompt :

  for (auto id : tokens_list) {
    std::cout << llama_token_to_str(ctx, id) << std::endl;
  }

  // Prediction loop

  std::vector<std::string> generated_tokens;

  while (llama_get_kv_cache_token_count(ctx) < max_context_size) {
    //---------------------------------
    // Evaluate the tokens :
    //---------------------------------

    if (llama_eval(ctx, tokens_list.data(), int(tokens_list.size()),
                   llama_get_kv_cache_token_count(ctx), params.n_threads)) {
      std::cout << "Evaluation Failed" << __func__ << std::endl;
      // return 1;
      // TODO: Raise exception here
    }

    // tokens_list.clear();

    //---------------------------------
    // Select the best prediction :
    //---------------------------------

    llama_token new_token_id = 0;

    auto logits = llama_get_logits(ctx);
    auto n_vocab =
        llama_n_vocab(ctx);  // the size of the LLM vocabulary (in tokens)

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(
          llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = {candidates.data(), candidates.size(),
                                           false};

    // Select it using the "Greedy sampling" method :
    new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

    // is it an end of stream ?
    if (new_token_id == llama_token_eos()) {
      // fprintf(stderr, " [end of text]\n");
      break;
    }

    generated_tokens.push_back(llama_token_to_str(ctx, new_token_id));

    // Print the new token :
    std::cout << llama_token_to_str(ctx, new_token_id) << std::endl;

    // Push this new token for next evaluation :
    tokens_list.push_back(new_token_id);

  }  // wend of main loop

  torch::Tensor tokens_tensor =
      torch::from_blob(tokens_list.data(),
                       {static_cast<long>(tokens_list.size())}, torch::kInt64);

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

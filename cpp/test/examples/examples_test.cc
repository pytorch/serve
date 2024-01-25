#include <fstream>

#include "test/utils/common.hh"

TEST_F(ModelPredictTest, TestLoadPredictBabyLlamaHandler) {
  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/examples/babyllama/babyllama_handler",
                        "llm", -1, "", "", 1, false),
                    "test/resources/examples/babyllama/babyllama_handler",
                    "test/resources/examples/babyllama/prompt.txt", "llm_ts",
                    200);
}

TEST_F(ModelPredictTest, TestLoadPredictLlmHandler) {
  std::ifstream f(
      "test/resources/examples/llamacpp/llamacpp_handler/"
      "llama-2-7b-chat.Q5_0.gguf");
  if (!f.good())
    GTEST_SKIP()
        << "Skipping TestLoadPredictLlmHandler because of missing file: "
           "test/resources/examples/llamacpp/llamacpp_handler/"
           "llama-2-7b-chat.Q5_0.gguf";

  this->LoadPredict(std::make_shared<torchserve::LoadModelRequest>(
                        "test/resources/examples/llamacpp/llamacpp_handler",
                        "llamacpp", -1, "", "", 1, false),
                    "test/resources/examples/llamacpp/llamacpp_handler",
                    "test/resources/examples/llamacpp/prompt.txt", "llm_ts",
                    200);
}

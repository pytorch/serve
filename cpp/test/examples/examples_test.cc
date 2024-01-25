#include <fstream>

#include "test/utils/common.hh"

TEST_F(ModelPredictTest, TestLoadPredictBabyLlamaHandler) {
  this->LoadPredict(
      std::make_shared<torchserve::LoadModelRequest>(
          "test/resources/torchscript_model/babyllama/babyllama_handler", "llm",
          -1, "", "", 1, false),
      "test/resources/torchscript_model/babyllama/babyllama_handler",
      "test/resources/torchscript_model/babyllama/prompt.txt", "llm_ts", 200);
}

TEST_F(ModelPredictTest, TestLoadPredictLlmHandler) {
  std::ifstream f(
      "test/resources/torchscript_model/llamacpp/llamacpp_handler/"
      "llama-2-7b-chat.Q5_0.gguf");
  if (!f.good())
    GTEST_SKIP()
        << "Skipping TestLoadPredictLlmHandler because of missing file: "
           "test/resources/torchscript_model/llamacpp/llamacpp_handler/"
           "llama-2-7b-chat.Q5_0.gguf";

  this->LoadPredict(
      std::make_shared<torchserve::LoadModelRequest>(
          "test/resources/torchscript_model/llamacpp/llamacpp_handler", "llm",
          -1, "", "", 1, false),
      "test/resources/torchscript_model/llamacpp/llamacpp_handler",
      "test/resources/torchscript_model/llamacpp/prompt.txt", "llm_ts", 200);
}

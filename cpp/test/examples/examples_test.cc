#include <fstream>

#include "test/utils/common.hh"

TEST_F(ModelPredictTest, TestLoadPredictBabyLlamaHandler) {
  std::string base_dir = "test/resources/examples/babyllama/";
  std::string file1 = base_dir + "babyllama_handler/stories15M.bin";
  std::string file2 = base_dir + "babyllama_handler/tokenizer.bin";

  std::ifstream f1(file1);
  std::ifstream f2(file2);

  if (!f1.good() && !f2.good())
    GTEST_SKIP()
        << "Skipping TestLoadPredictBabyLlamaHandler because of missing files: "
        << file1 << " or " << file2;

  this->LoadPredict(
      std::make_shared<torchserve::LoadModelRequest>(
          base_dir + "babyllama_handler", "llm", -1, "", "", 1, false),
      base_dir + "babyllama_handler", base_dir + "prompt.txt", "llm_ts", 200);
}

TEST_F(ModelPredictTest, TestLoadPredictLlmHandler) {
  std::string base_dir = "test/resources/examples/llamacpp/";
  std::string file1 = base_dir + "llamacpp_handler/llama-2-7b-chat.Q5_0.gguf";
  std::ifstream f(file1);

  if (!f.good())
    GTEST_SKIP()
        << "Skipping TestLoadPredictLlmHandler because of missing file: "
        << file1;

  this->LoadPredict(
      std::make_shared<torchserve::LoadModelRequest>(
          base_dir + "llamacpp_handler", "llamacpp", -1, "", "", 1, false),
      base_dir + "llamacpp_handler", base_dir + "prompt.txt", "llm_ts", 200);
}

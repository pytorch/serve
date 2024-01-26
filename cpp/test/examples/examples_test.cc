#include "test/utils/common.hh"

TEST_F(ModelPredictTest, TestLoadPredictBabyLlamaHandler) {
  this->LoadPredict(
      std::make_shared<torchserve::LoadModelRequest>(
          "test/resources/torchscript_model/babyllama/babyllama_handler", "llm",
          -1, "", "", 1, false),
      "test/resources/torchscript_model/babyllama/babyllama_handler",
      "test/resources/torchscript_model/babyllama/prompt.txt", "llm_ts", 200);
}

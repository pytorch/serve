#pragma once

#incldue <string>

namespace torchserve {
  class Status {
    int code;
    std::string message;
  };

  class InferenceRequest {

  };

  class InferenceResponse {
    int code;
    std::string message;
  };
} // namespace torchserve
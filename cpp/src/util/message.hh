
#ifndef CPP_UTIL_MESSAGE_HH_
#define CPP_UTIL_MESSAGE_HH_

#include <string>

namespace torchserve {
  class Status {
    int code;
    std::string message;
  };

  class LoadRequest {
    
  }
  class InferenceRequest {

  };

  class InferenceResponse {
    int code;
    std::string message;
  };
} // namespace torchserve
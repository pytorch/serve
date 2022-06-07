#pragma once

#include <map>
#include <ostream>

namespace torchserve {
  enum RuntimeType {
    PYTHON,
    PYTHON2,
    PYTHON3, 
    LDP,
    LSP,
  };

  const std::map<std::string, RuntimeType> runtimeType_mapping = {
      {"python", PYTHON},
      {"python2", PYTHON2},
      {"python3", PYTHON3},
      {"libtorch_deploy_process", LDP},
      {"libtorch_scripted_process", LSP}
    };

  RuntimeType get_runtime_type_from_string(const std::string &type_lower_case) {
    std::map<std::string, RuntimeType>::const_iterator it = runtimeType_mapping.find(type_lower_case);
    
    if (it == runtimeType_mapping.end()) {
      // logger error
    }
    return it->second;
  }
}

#pragma once

#include <map>
#include <memory>

#include "src/backends/handler/base_handler.hh"
#include "src/backends/handler/torch_scripted_handler.hh"

namespace torchserve {
class HandlerFactory {
 public:
  static HandlerFactory GetInstance() {
    static HandlerFactory instance;
    return instance;
  };

  std::shared_ptr<BaseHandler> createHandler(
      const std::string& handler_class_name) {
    auto it = handlers_.find(handler_class_name);
    if (it == handlers_.end()) {
      return std::shared_ptr<BaseHandler>(nullptr);
    } else {
      return it->second();
    }
  };

 private:
  std::map<std::string, std::shared_ptr<BaseHandler> (*)()> handlers_ = {
      {"TorchScriptHandler", []() -> std::shared_ptr<BaseHandler> {
         return std::make_shared<TorchScriptHandler>();
       }}};
  HandlerFactory(){};
};
}  // namespace torchserve

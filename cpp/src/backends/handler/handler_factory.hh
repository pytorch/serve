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
      // XXX:
      // Why not use the default ctor of `std::shared_ptr` directly?
      // What are the benefits of using this `std::shared_ptr(nullptr_t)`?
      return std::shared_ptr<BaseHandler>(nullptr);
    } else {
      return it->second();
    }
  };

 private:
  // XXX:
  // 1) What are the benefits of using a function (ctor) pointer as the value
  // instead of using a `shared_ptr` instance directly?
  // 2) Whenever we want to add a new pair to `handlers_`, we'll have to
  // change the definition here.
  std::map<std::string, std::shared_ptr<BaseHandler> (*)()> handlers_ = {
      {"TorchScriptHandler", []() -> std::shared_ptr<BaseHandler> {
         return std::make_shared<TorchScriptHandler>();
       }}};
  HandlerFactory(){};
};
}  // namespace torchserve

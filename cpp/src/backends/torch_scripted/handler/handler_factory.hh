#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_HANDLER_FACTORY_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_HANDLER_FACTORY_HH_

#include <map>
#include <memory>

#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace torchserve {
namespace torchscripted {
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
      {"BaseHandler", []() -> std::shared_ptr<BaseHandler> {
         return std::make_shared<BaseHandler>();
       }}};
  HandlerFactory(){};
};
}  // namespace torchscripted
}  // namespace torchserve
#endif  // TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_HANDLER_FACTORY_HH_
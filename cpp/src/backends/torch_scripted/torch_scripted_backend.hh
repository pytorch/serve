#pragma once

#include <fmt/format.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <memory>

#include "src/backends/core/backend.hh"
#include "src/backends/handler/base_handler.hh"
#include "src/backends/torch_scripted/handler/handler_factory.hh"
#include "src/utils/dl_loader.hh"
#include "src/utils/logging.hh"
#include "src/utils/message.hh"
#include "src/utils/model_archive.hh"

namespace torchserve {
namespace torchscripted {
class Backend final : public torchserve::Backend {
 public:
  Backend() = default;
  ~Backend() override {
    if (dl_loader_ && handler_) {
      handler_.reset();
    }
  };

  bool Initialize(const std::string& model_dir) override;

  std::unique_ptr<torchserve::LoadModelResponse> LoadModelInternal(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request)
      override;

 private:
  void LoadHandler(const std::string& model_dir);

  std::unique_ptr<torchserve::DLLoader<torchserve::BaseHandler>> dl_loader_;
  std::shared_ptr<torchserve::BaseHandler> handler_;
};
}  // namespace torchscripted
}  // namespace torchserve

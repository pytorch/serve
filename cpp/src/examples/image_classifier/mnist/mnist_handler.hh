#ifndef MNIST_HANDLER_HH_
#define MNIST_HANDLER_HH_

#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace mnist {
  class MnistHandler : public torchserve::torchscripted::BaseHandler {
    public:
    MnistHandler() = default;
    ~MnistHandler() = default;

    void Postprocess(
      const torch::Tensor& data,
      std::map<uint8_t, std::string>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) override;
  };
} // namespace mnist
#endif // MNIST_HANDLER_HH_
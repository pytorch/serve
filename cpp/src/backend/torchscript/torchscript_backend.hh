#pragma once

#include "backend.hh"

namespace torchserve {
class TorchscriptBackend : public Backend {};

class TorchscriptBackendModel : public BackendModel {};
}  // namespace torchserve
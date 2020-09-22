#include <torch/extension.h>
#include <torch/script.h>
#include <vector>



/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
torch::jit::script::Module load_model(const std::string &fpath, const std::string& map_location, const std::string& device){
   torch::jit::script::Module model = torch::jit::load(fpath, map_location);
   model.to(device);
   model.eval();
   return model;
}


/// Call the forward pass on loaded model.
///
torch::Tensor run_model(torch::jit::script::Module model, std::vector<torch::Tensor> input_tensors) {
  std::vector<torch::jit::IValue> inputs;
  for (auto& t : input_tensors) {
    inputs.push_back(t);
  }
  return model.forward(inputs).toTensor();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = R"pbdoc(
         Torch CPP Torch Script model loading and inference API bindings
        -------------------------------------------------------------

        .. currentmodule:: TORCH_EXTENSION_NAME

        .. autosummary::
           :toctree: _generate

           load_model
           run_model

    )pbdoc";

  m.def("load_model", &load_model,  R"pbdoc(
        Load Torch Script model
        Accepts path to the model and device name.
    )pbdoc");

  m.def("run_model", &run_model,  R"pbdoc(
        Do the inference on loaded Torch Script model
        Accepts a model and list/vector of Torch Tensors.
    )pbdoc");

}



#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <numeric>
#include <iomanip>

namespace py = pybind11;

class Singleton {
private:
    static bool instanceFlag;
    static Singleton *single;
    torch::jit::script::Module module;

    Singleton(const std::string &fname) {
          try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(fname);
          }
          catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
          }

    }

public:
    static Singleton *getInstance();

    static Singleton *getInstance(const std::string &fname);

    std::vector<float> run_model(std::vector<float> input_data,  std::vector<int64_t> input_data_shape);

    ~Singleton() {
        instanceFlag = false;
    }
};

bool Singleton::instanceFlag = false;
Singleton *Singleton::single = NULL;

Singleton *Singleton::getInstance(const std::string &fname) {
    if (!instanceFlag) {
        single = new Singleton(fname);
        instanceFlag = true;
        return single;
    } else {
        return single;
    }
}

Singleton *Singleton::getInstance() {
    return single;
}

std::vector<float> Singleton::run_model(std::vector<float> input_data, std::vector<int64_t> input_data_shape) {
  // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(torch::ones({1, 3, 224, 224}));

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor tharray = torch::from_blob(input_data.data(), input_data_shape);

    inputs.push_back(tharray);


    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    at::Tensor my_tensor = output.slice(/*dim=*/1, /*start=*/0, /*end=*/5);
    std::vector<float> v(my_tensor.data<float>(), my_tensor.data<float>() + my_tensor.numel());
    return v;
}



void load_model(const std::string &fname) {
    Singleton *sc1;
    sc1 = Singleton::getInstance(fname);
}

std::vector<float> run_model(std::vector<float> input_data, std::vector<int64_t> input_data_shape) {
    Singleton *sc2;
    sc2 = Singleton::getInstance();
    return sc2->run_model(input_data, input_data_shape);
}


PYBIND11_MODULE(pt_c_inference, m) {
    m.doc() = R"pbdoc(
        Pybnind module to invoke Pytorch C++ infernce
        -----------------------

        .. currentmodule:: pt_c_inference

        .. autosummary::
           :toctree: _generate

           load_model
           run_model
    )pbdoc";


    //m.def("run1", &run1, "Run inference");
    m.def("load_model", &load_model, "Load Model");
    m.def("run_model", &run_model, "Run inference");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}




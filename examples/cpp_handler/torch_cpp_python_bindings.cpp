#include <iostream>
#include <vector>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
torch::jit::script::Module initialize(const std::string &fpath, const std::string& map_location,
 const std::string& device){
   torch::jit::script::Module model = torch::jit::load(fpath, map_location);
   model.to(device);
   model.eval();
   return model;
}

/// Pre-process the raw input image
/// Call the forward pass on loaded model.
///
std::tuple<at::Tensor, at::Tensor>  handle(torch::jit::script::Module model, const std::string raw,
const std::string& device, const int topk){

    std::vector<at::Tensor> img_tensors;
    std::vector<std::uint8_t> vectordata(raw.begin(),raw.end());
    cv::Mat data_mat = cv::Mat(vectordata, false);
    cv::Mat img(cv::imdecode(data_mat, 1));

    cv::cvtColor( img, img, cv::COLOR_BGR2RGB );
    cv::Size rsz = { 224, 224 };

    cv::resize( img, img, rsz, 0, 0, cv::INTER_LINEAR );
    img.convertTo( img, CV_32FC3, 1/255.0 );

    at::Tensor tensorImage = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 }, at::kFloat);
    tensorImage = tensorImage.permute({ 0, 3, 1, 2 });

    //  Normalize data
    tensorImage[0][0] = tensorImage[0][0].sub(0.485).div(0.229);
    tensorImage[0][1] = tensorImage[0][1].sub(0.456).div(0.224);
    tensorImage[0][2] = tensorImage[0][2].sub(0.406).div(0.225);

    tensorImage.to(device);

    std::vector<torch::jit::IValue> input;
    input.push_back(tensorImage);

    at::Tensor out = model.forward(input).toTensor();

    at::Tensor output = torch::softmax(out, 1);
    std::tuple<at::Tensor, at::Tensor> result = torch::topk(output, topk, 1);

    return result;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = R"pbdoc(
         Torch CPP Torch Script model loading and inference API bindings
        -------------------------------------------------------------

        .. currentmodule:: TORCH_EXTENSION_NAME

        .. autosummary::
           :toctree: _generate

           initialize
           handle

    )pbdoc";


  m.def("initialize", &initialize,  R"pbdoc(
        Load Torch Script model
        Accepts path to the model and device name.
    )pbdoc");

  m.def("handle", &handle,  R"pbdoc(
        Do the inference on loaded Torch Script model
        Accepts a model, input raw data, device, topk value.
    )pbdoc");

}




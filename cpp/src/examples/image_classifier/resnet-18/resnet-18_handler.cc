#include "src/examples/image_classifier/resnet-18/resnet-18_handler.hh"

#include <folly/json.h>

#include <fstream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

namespace resnet {

constexpr int kTargetImageSize = 224;
constexpr double kImageNormalizationMeanR = 0.485;
constexpr double kImageNormalizationMeanG = 0.456;
constexpr double kImageNormalizationMeanB = 0.406;
constexpr double kImageNormalizationStdR = 0.229;
constexpr double kImageNormalizationStdG = 0.224;
constexpr double kImageNormalizationStdB = 0.225;
constexpr int kTopKClasses = 5;

std::vector<torch::jit::IValue> ResnetHandler::Preprocess(
    std::shared_ptr<torch::Device>& device,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  std::vector<torch::jit::IValue> batch_ivalue;
  std::vector<torch::Tensor> batch_tensors;
  uint8_t idx = 0;
  for (auto& request : *request_batch) {
    (*response_batch)[request.request_id] =
        std::make_shared<torchserve::InferenceResponse>(request.request_id);
    idx_to_req_id.first += idx_to_req_id.first.empty()
                               ? request.request_id
                               : "," + request.request_id;
    auto data_it =
        request.parameters.find(torchserve::PayloadType::kPARAMETER_NAME_DATA);
    auto dtype_it =
        request.headers.find(torchserve::PayloadType::kHEADER_NAME_DATA_TYPE);
    if (data_it == request.parameters.end()) {
      data_it = request.parameters.find(
          torchserve::PayloadType::kPARAMETER_NAME_BODY);
      dtype_it =
          request.headers.find(torchserve::PayloadType::kHEADER_NAME_BODY_TYPE);
    }

    if (data_it == request.parameters.end() ||
        dtype_it == request.headers.end()) {
      TS_LOGF(ERROR, "Empty payload for request id: {}", request.request_id);
      (*response_batch)[request.request_id]->SetResponse(
          500, "data_type", torchserve::PayloadType::kCONTENT_TYPE_TEXT,
          "Empty payload");
      continue;
    }

    try {
      if (dtype_it->second == torchserve::PayloadType::kDATA_TYPE_BYTES) {
        cv::Mat image = cv::imdecode(data_it->second, cv::IMREAD_COLOR);

        // Check if the image was successfully decoded
        if (image.empty()) {
          std::cerr << "Failed to decode the image.\n";
        }

        const int rows = image.rows;
        const int cols = image.cols;

        const int cropSize = std::min(rows, cols);
        const int offsetW = (cols - cropSize) / 2;
        const int offsetH = (rows - cropSize) / 2;

        const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
        image = image(roi);

        // Convert the image to GPU Mat
        cv::cuda::GpuMat gpuImage;
        cv::Mat resultImage;

        gpuImage.upload(image);

        // Resize on GPU
        cv::cuda::resize(gpuImage, gpuImage,
                         cv::Size(kTargetImageSize, kTargetImageSize));

        // Convert to BGR on GPU
        cv::cuda::cvtColor(gpuImage, gpuImage, cv::COLOR_BGR2RGB);

        // Convert to float on GPU
        gpuImage.convertTo(gpuImage, CV_32FC3, 1 / 255.0);

        // Download the final image from GPU to CPU
        gpuImage.download(resultImage);

        // Create a tensor from the CPU Mat
        torch::Tensor tensorImage = torch::from_blob(
            resultImage.data, {resultImage.rows, resultImage.cols, 3},
            torch::kFloat);
        tensorImage = tensorImage.permute({2, 0, 1});

        std::vector<double> norm_mean = {kImageNormalizationMeanR,
                                         kImageNormalizationMeanG,
                                         kImageNormalizationMeanB};
        std::vector<double> norm_std = {kImageNormalizationStdR,
                                        kImageNormalizationStdG,
                                        kImageNormalizationStdB};

        // Normalize the tensor
        tensorImage = torch::data::transforms::Normalize<>(
            norm_mean, norm_std)(tensorImage);

        tensorImage.clone();
        batch_tensors.emplace_back(tensorImage.to(*device));
        idx_to_req_id.second[idx++] = request.request_id;
      } else if (dtype_it->second == "List") {
        // case3: the image is a list
      }
    } catch (const std::runtime_error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, error: {}",
              request.request_id, e.what());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
    } catch (const c10::Error& e) {
      TS_LOGF(ERROR, "Failed to load tensor for request id: {}, c10 error: {}",
              request.request_id, e.msg());
      auto response = (*response_batch)[request.request_id];
      response->SetResponse(500, "data_type",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
    }
  }
  if (!batch_tensors.empty()) {
    batch_ivalue.emplace_back(torch::stack(batch_tensors).to(*device));
  }

  return batch_ivalue;
}

void ResnetHandler::Postprocess(
    const torch::Tensor& data,
    std::pair<std::string&, std::map<uint8_t, std::string>&>& idx_to_req_id,
    std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
  for (const auto& kv : idx_to_req_id.second) {
    try {
      auto response = (*response_batch)[kv.second];
      namespace F = torch::nn::functional;

      // Perform softmax and top-k operations
      torch::Tensor ps = F::softmax(data, F::SoftmaxFuncOptions(1));
      std::tuple<torch::Tensor, torch::Tensor> result =
          torch::topk(ps, kTopKClasses, 1, true, true);
      torch::Tensor probs = std::get<0>(result);
      torch::Tensor classes = std::get<1>(result);

      probs = probs.to(torch::kCPU);
      classes = classes.to(torch::kCPU);
      // Convert tensors to C++ vectors
      std::vector<float> probs_vector(probs.data_ptr<float>(),
                                      probs.data_ptr<float>() + probs.numel());
      std::vector<long> classes_vector(
          classes.data_ptr<long>(), classes.data_ptr<long>() + classes.numel());

      // Create a JSON object using folly::dynamic
      folly::dynamic json_response = folly::dynamic::object;
      // Create a folly::dynamic array to hold tensor elements
      folly::dynamic probability = folly::dynamic::array;
      folly::dynamic class_names = folly::dynamic::array;

      // Iterate through tensor elements and add them to the dynamic_array
      for (const float& value : probs_vector) {
        probability.push_back(value);
      }
      for (const long& value : classes_vector) {
        class_names.push_back(value);
      }
      // Add key-value pairs to the JSON object
      json_response["probability"] = probability;
      json_response["classes"] = class_names;

      // Serialize the JSON object to a string
      std::string json_str = folly::toJson(json_response);

      // Serialize and set the response
      response->SetResponse(200, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_BYTES,
                            json_str);
    } catch (const std::runtime_error& e) {
      LOG(ERROR) << "Failed to load tensor for request id:" << kv.second
                 << ", error: " << e.what();
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "runtime_error, failed to load tensor");
      throw e;
    } catch (const c10::Error& e) {
      LOG(ERROR) << "Failed to load tensor for request id:" << kv.second
                 << ", c10 error: " << e.msg();
      auto response = (*response_batch)[kv.second];
      response->SetResponse(500, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_STRING,
                            "c10 error, failed to load tensor");
      throw e;
    }
  }
}

}  // namespace resnet

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::torchscripted::BaseHandler* allocatorResnetHandler() {
  return new resnet::ResnetHandler();
}

void deleterResnetHandler(torchserve::torchscripted::BaseHandler* p) {
  if (p != nullptr) {
    delete static_cast<resnet::ResnetHandler*>(p);
  }
}
}
#endif

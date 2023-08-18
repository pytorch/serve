#include "src/examples/image_classifier/resnet-18/resnet-18_handler.hh"

#include <folly/json.h>
#include <fstream>

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
    /*
    case2: the image is sent as string of bytesarray
    if (dtype_it->second == "String") {
      try {
        auto b64decoded_str = folly::base64Decode(data_it->second);
        torchserve::Converter::StrToBytes(b64decoded_str, image);
      } catch (folly::base64_decode_error e) {
        TS_LOGF(ERROR, "Failed to base64Decode for request id: {}, error: {}",
                request.request_id,
                e.what());
      }
    }
    */

    try {
      if (dtype_it->second == torchserve::PayloadType::kDATA_TYPE_BYTES) {
        // case2: the image is sent as bytesarray
        // torch::serialize::InputArchive archive;
        // archive.load_from(std::istringstream
        // iss(std::string(data_it->second)));
        /*
        std::istringstream iss(std::string(data_it->second.begin(),
        data_it->second.end())); torch::serialize::InputArchive archive;
        images.emplace_back(archive.load_from(iss, torch::Device device);

        std::vector<char> bytes(
          static_cast<char>(*data_it->second.begin()),
          static_cast<char>(*data_it->second.end()));

        images.emplace_back(torch::pickle_load(bytes).toTensor().to(*device));
        */

        cv::Mat image = cv::imdecode(data_it->second, cv::IMREAD_COLOR);

        // Check if the image was successfully decoded
        if (image.empty()) {
          std::cerr << "Failed to decode the image.\n";
        }

        // Crop image
        const int rows = image.rows;
        const int cols = image.cols;

        const int cropSize = std::min(rows, cols);
        const int offsetW = (cols - cropSize) / 2;
        const int offsetH = (rows - cropSize) / 2;

        const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
        image = image(roi);

        // Resize
        cv::resize(image, image, cv::Size(kTargetImageSize, kTargetImageSize));

        // Convert BGR to RGB format
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        image.convertTo(image, CV_32FC3, 1 / 255.0);

        // Convert the OpenCV image to a torch tensor
        torch::Tensor tensorImage = torch::from_blob(
            image.data, {image.rows, image.cols, 3}, c10::kFloat);
        tensorImage = tensorImage.permute({2, 0, 1});

        // Normalize
        std::vector<double> norm_mean = {kImageNormalizationMeanR,
                                         kImageNormalizationMeanG,
                                         kImageNormalizationMeanB};
        std::vector<double> norm_std = {kImageNormalizationStdR,
                                        kImageNormalizationStdG,
                                        kImageNormalizationStdB};

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
  std::ifstream jsonFile("index_to_name.json");
  if (!jsonFile.is_open()) {
      std::cerr << "Failed to open JSON file.\n";
      return 1;
  }
  std::string jsonString((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
  jsonFile.close();
  folly::dynamic parsedJson = folly::parseJson(jsonString);
  if (!parsedJson.isObject()) {
      std::cerr << "Invalid JSON format.\n";
      return 1;
  }
  for (const auto& kv : idx_to_req_id.second) {
    try {
      auto response = (*response_batch)[kv.second];
      namespace F = torch::nn::functional;

      // Perform softmax and top-k operations
      torch::Tensor ps = F::softmax(data, F::SoftmaxFuncOptions(1));
      std::tuple<torch::Tensor, torch::Tensor> result =
          torch::topk(ps, kTopKClasses, 1, true, true);
      auto [probs, classes] = result;

      // Convert tensors to C++ vectors
      std::vector<float> probs_vector(probs.data<float>(),
                                      probs.data<float>() + probs.numel());
      std::vector<long> classes_vector(classes.data<long>(),
                                       classes.data<long>() + classes.numel());

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

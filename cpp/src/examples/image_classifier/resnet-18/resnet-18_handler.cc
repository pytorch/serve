#include "src/examples/image_classifier/resnet-18/resnet-18_handler.hh"

#include <opencv2/opencv.hpp>

namespace resnet {

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

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Check if the image was successfully decoded
        if (image.empty()) {
          std::cerr << "Failed to decode the image." << std::endl;
        }

        // Resize
        const int newWidth = 256, newHeight = 256;
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

        // Crop image
        const int cropSize = 224;
        const int offsetW = (resizedImage.cols - cropSize) / 2;
        const int offsetH = (resizedImage.rows - cropSize) / 2;

        const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
        cv::Mat croppedImage = resizedImage(roi).clone();

        // Convert the OpenCV image to a torch tensor
        // Drift in cropped image
        // Vision Crop: 114, 118, 115, 102, 106, 97
        // OpenCV Crop: 113, 118, 114, 100, 106, 97
        torch::TensorOptions options(torch::kByte);
        torch::Tensor tensorImage = torch::from_blob(
            croppedImage.data,
            {croppedImage.rows, croppedImage.cols, croppedImage.channels()},
            options);

        tensorImage = tensorImage.permute({2, 0, 1});
        tensorImage = tensorImage.to(torch::kFloat32) / 255.0;

        // Normalize
        torch::Tensor normalizedTensorImage =
            torch::data::transforms::Normalize<>(
                {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(tensorImage);
        normalizedTensorImage.clone();
        batch_tensors.emplace_back(normalizedTensorImage.to(*device));
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
      torch::Tensor ps = F::softmax(data, F::SoftmaxFuncOptions(1));
      std::tuple<torch::Tensor, torch::Tensor> result =
          torch::topk(ps, 5, 1, true, true);
      auto [probs, classes] = result;
      // tensor([[0.4097, 0.3467, 0.1300, 0.0239, 0.0115]]) tensor([[281, 282,
      // 285, 287, 463]])
      response->SetResponse(200, "data_tpye",
                            torchserve::PayloadType::kDATA_TYPE_BYTES,
                            torch::pickle_save(probs[kv.first]));
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

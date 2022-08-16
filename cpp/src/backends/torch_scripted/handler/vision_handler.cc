#include "src/backends/torch_scripted/handler/vision_handler.hh"

namespace torchserve {
  namespace torchscripted {
    std::vector<torch::jit::IValue> VisionHandler::Preprocess(
      std::shared_ptr<torch::Device>& device,
      std::map<uint8_t, std::string>& idx_to_req_i,
      std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch)  {
      /**
       * @brief 
       * Ref: https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py#L27
       */
      std::vector<torch::jit::IValue> images;
      uint8_t idx = 0;
      for (auto& request : *request_batch) {
        auto data_it = request.parameters.find("data");
        auto dtype_it = request.headers.find("data_dtype");
        if (data_it == request.parameters.end()) {
          data_it = request.parameters.find("body");
          dtype_it = request.headers.find("body_dtype");
        }

        if (data_it == request.parameters.end() || 
          dtype_it == request.headers.end()) {
          LOG(ERROR) << "Empty payload for request id:" << request.request_id;
          auto response = (*response_batch)[request.request_id];
          response->SetResponse(500, "data_tpye", torchserve::CONTENT_TYPE_TEXT, "Empty payload");
          continue;
        } 
        /*
        case2: the image is sent as string of bytesarray
        if (dtype_it->second == "String") {
          try {
            auto b64decoded_str = folly::base64Decode(data_it->second);
            torchserve::Converter::StrToBytes(b64decoded_str, image);
          } catch (folly::base64_decode_error e) {
            LOG(ERROR) << "Failed to base64Decode for request id:" << request.request_id 
            << ", error: " << e.what();
          }
        }
        */

        try {
          if (dtype_it->second == "Bytes") {
            // case2: the image is sent as bytesarray
            //torch::serialize::InputArchive archive;
            //archive.load_from(std::istringstream iss(std::string(data_it->second)));
            /*
            std::istringstream iss(std::string(data_it->second.begin(), data_it->second.end()));
            torch::serialize::InputArchive archive;
            images.emplace_back(archive.load_from(iss, torch::Device device);
            */
            std::vector<char> bytes(
              static_cast<char>data_it->second.begin(), 
              static_cast<char>data_it->second.end());
            images.emplace_back(torch::pickle_load(bytes).toTensor().to(*device));
            idx_to_req_i[idx++] = request.request_id;
          } else if (dtype_it->second == "List") {
            // case3: the image is a list
          }
        } catch (std::runtime_error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << request.request_id 
          << ", error: " << e.what();
          auto response = (*response_batch)[request.request_id];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::DATA_TYPE_STRING,
            "runtime_error, failed to load tensor");
        } catch (const c10::Error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << request.request_id 
          << ", c10 error: " << e.msg();
          auto response = (*response_batch)[request.request_id];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::DATA_TYPE_STRING,
            "c10 error, failed to load tensor");
        }
      }
      return images;
    }

    torch::Tensor VisionHandler::Predict(
      std::shared_ptr<torch::jit::script::Module> model, 
      torch::Tensor& inputs,
      std::shared_ptr<torch::Device>& device,
      std::map<uint8_t, std::string>& idx_to_req_i,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
      try {
        torch::NoGradGuard no_grad;
        return model->forward(inputs).toTensor();
      } catch std::runtime_error& e) {
        LOG(ERROR) << "Failed to predict, error:" << e.what();
        uint8_t idx = 0;
        for (auto& kv : idx_to_req_id) {
          auto response = (*response_batch)[kv.second];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::DATA_TYPE_STRING,
            "runtime_error, failed to predict");
        }
      }
    }

    void VisionHandler::Postprocess(
      const torch::Tensor& data,
      std::map<uint8_t, std::string>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
      for (const auto& kv : idx_to_req_id) {
        try {
          std::ostringstream oss;
          torch::save(data[kv.first], oss);
          auto response = (*response_batch)[kv.second];
          response->SetResponse(
            200,
            "data_tpye",
            torchserve::DATA_TYPE_BYTES,
            oss.str());
            oss.close();
          );
        } catch (std::runtime_error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << request.request_id 
          << ", error: " << e.what();
          auto response = (*response_batch)[request.request_id];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::DATA_TYPE_STRING,
            "runtime_error, failed to load tensor");
          } catch (const c10::Error& e) {
            LOG(ERROR) << "Failed to load tensor for request id:" << request.request_id 
            << ", c10 error: " << e.msg();
            auto response = response_batch->req_id_to_response[request.request_id];
            response->SetResponse(
              500, 
              "data_tpye", 
             torchserve::DATA_TYPE_STRING,
              "c10 error, failed to load tensor");
          }
        }
      }
    }
  } // namespace torchscripted
} // namespace torchserve 
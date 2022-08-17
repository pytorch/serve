#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace torchserve {
  namespace torchscripted {
    std::pair<std::shared_ptr<torch::jit::script::Module>, std::shared_ptr<torch::Device>> 
    BaseHandler::LoadModel(
      std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
      try {
        auto device = GetTorchDevice(load_model_request);
        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(
          // TODO: windows
          fmt::format("{}/{}", 
          load_model_request->model_dir, 
          manifest_->GetModel().serialized_file),
          *device));
        return std::make_pair(module, device);
      } catch (const c10::Error& e) {
        LOG(ERROR) << "loading the model: " 
        << load_model_request->model_name 
        << ", device id:" 
        << load_model_request->gpu_id
        << ", error: " << e.msg();
        throw e;
      } catch (const std::runtime_error& e) {
        LOG(ERROR) << "loading the model: " 
        << load_model_request->model_name 
        << ", device id:" 
        << load_model_request->gpu_id
        << ", error1: " << e.what();
        throw e;
      }
    }

    std::shared_ptr<torch::Device> BaseHandler::GetTorchDevice(
      std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
      /**
       * @brief 
       * TODO: extend LoadModelRequest to support 
       * - device type: CPU, GPU or others
       */
      if (load_model_request->gpu_id < 0) {
        return std::make_shared<torch::Device>(torch::kCPU);
      } 

      return std::make_shared<torch::Device>(torch::kCUDA, load_model_request->gpu_id);
    }

    std::vector<torch::jit::IValue> BaseHandler::Preprocess(
      std::shared_ptr<torch::Device>& device,
      std::map<uint8_t, std::string>& idx_to_req_id,
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
          response->SetResponse(500, "data_tpye", torchserve::PayloadType::kCONTENT_TYPE_TEXT, "Empty payload");
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
              static_cast<char>(*data_it->second.begin()), 
              static_cast<char>(*data_it->second.end()));
            images.emplace_back(torch::pickle_load(bytes).toTensor().to(*device));
            idx_to_req_id[idx++] = request.request_id;
          } else if (dtype_it->second == "List") {
            // case3: the image is a list
          }
        } catch (const std::runtime_error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << request.request_id 
          << ", error: " << e.what();
          auto response = (*response_batch)[request.request_id];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::PayloadType::kDATA_TYPE_STRING,
            "runtime_error, failed to load tensor");
            throw e;
        } catch (const c10::Error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << request.request_id 
          << ", c10 error: " << e.msg();
          auto response = (*response_batch)[request.request_id];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::PayloadType::kDATA_TYPE_STRING,
            "c10 error, failed to load tensor");
          throw e;
        }
      }
      return images;
    }

    torch::Tensor BaseHandler::Predict(
      std::shared_ptr<torch::jit::script::Module> model, 
      std::vector<torch::jit::IValue>& inputs,
      //std::shared_ptr<torch::Device>& device,
      std::map<uint8_t, std::string>& idx_to_req_id,
      std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) {
      try {
        torch::NoGradGuard no_grad;
        return model->forward(inputs).toTensor();
      } catch (const std::runtime_error& e) {
        LOG(ERROR) << "Failed to predict, error:" << e.what();
        for (auto& kv : idx_to_req_id) {
          auto response = (*response_batch)[kv.second];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::PayloadType::kDATA_TYPE_STRING,
            "runtime_error, failed to predict");
        }
        throw e;
      }
    }

    void BaseHandler::Postprocess(
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
            torchserve::PayloadType::kDATA_TYPE_BYTES,
            oss.str()
          );
        } catch (const std::runtime_error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << kv.second  
          << ", error: " << e.what();
          auto response = (*response_batch)[kv.second];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::PayloadType::kDATA_TYPE_STRING,
            "runtime_error, failed to load tensor");
            throw e;
        } catch (const c10::Error& e) {
          LOG(ERROR) << "Failed to load tensor for request id:" << kv.second 
          << ", c10 error: " << e.msg();
          auto response = (*response_batch)[kv.second];
          response->SetResponse(
            500, 
            "data_tpye", 
            torchserve::PayloadType::kDATA_TYPE_STRING,
            "c10 error, failed to load tensor");
          throw e;
        }
      }
    }
  } // namespace torchscripted
} // namespace torchserve

/**
#if defined(__linux__) || defined(__APPLE__)
extern "C" {
  torchserve::torchscripted::BaseHandler *allocatorBaseHandler() {
    return new torchserve::torchscripted::BaseHandler();
  }
  
  void deleterBaseHandler(torchserve::torchscripted::BaseHandler *p) {
    delete p;
  }
}
#endif
*/
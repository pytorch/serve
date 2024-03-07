#include "json.hh"

#include <fstream>
#include <exception>

#include "logging.hh"



using json = nlohmann::json;

namespace torchserve
{
    Json::Json(nlohmann::json _data):data(_data){}

    Json::~Json(){}

    Json Json::ParseJsonFile(const std::string& filename)
    {
        try{
            std::ifstream fs(filename);
             auto data = json::parse(fs);
             return Json(data);
        }catch(std::exception e)
        {
            TS_LOGF(ERROR, "Error parsing json file: {} reason {}", filename, e.what());
        }
    }

    std::string Json::GetValueAsString(const std::string& key)
    {
        if(data.contains(key)){
            return data[key].template get<std::string>();
        }else{
            TS_LOGF(ERROR, "Key not found: {}", key);
            throw std::invalid_argument("Key not found: " + key);
        }
    }

    std::string Json::GetValueAsString(const int key)
    {
        if(key < data.size()){
            return data[key].template get<std::string>();
        }else{
            TS_LOGF(ERROR, "Key not found: {}", key);
            throw std::invalid_argument("Key not found: " + std::to_string(key));
        }
    }

    int Json::GetValueAsInt(const std::string& key)
    {
        if(data.contains(key)){
            return data[key].template get<int>();
        }else{
            TS_LOGF(ERROR, "Key not found: {}", key);
            throw std::invalid_argument("Key not found: " + key);
        }
    }

    int Json::GetValueAsInt(const int key)
    {
        if(key < data.size()){
            return data[key].template get<int>();
        }else{
            TS_LOGF(ERROR, "Key not found: {}", key);
            throw std::invalid_argument("Key not found: " + std::to_string(key));
        }
    }

    Json Json::GetValue(const std::string& key)
    {
        if(data.contains(key)){
            return Json(data[key]);
        }else{
            TS_LOGF(ERROR, "Key not found: {}", key);
            throw std::invalid_argument("Key not found: " + key);
        }
    }

    bool Json::HasKey(const std::string& key){
        return data.contains(key);
    }
}// namespace torchserve

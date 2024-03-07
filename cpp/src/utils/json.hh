#pragma once
#include <string>

#include <nlohmann/json.hpp>

namespace torchserve
{
    class Json{
    public:
        virtual ~Json();
        static Json ParseJsonFile(const std::string& filename);
        std::string GetValueAsString(const std::string& key);
        std::string GetValueAsString(const int key);
        int GetValueAsInt(const std::string& key);
        int GetValueAsInt(const int key);
        Json GetValue(const std::string& key);
        bool HasKey(const std::string& key);
    protected:
        Json(nlohmann::json _data);
        nlohmann::json data;
    };
} // namespace torchserve

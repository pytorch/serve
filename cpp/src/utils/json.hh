#pragma once
#include <string>

#include <nlohmann/json.hpp>

namespace torchserve
{
    class Json{
    public:
        virtual ~Json();
        static Json ParseJsonFile(const std::string& filename);
        Json GetValue(const std::string& key);
        Json GetValue(const unsigned long key);
        bool HasKey(const std::string& key);
        std::string AsString();
        int AsInt();
    protected:
        Json(nlohmann::json _data);
        nlohmann::json data;
    };
} // namespace torchserve

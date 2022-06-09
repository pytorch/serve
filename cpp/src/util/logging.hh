#pragma once

#include <string>

namespace torchserve {
  class Logger {
    public:

    explicit Logger(std::string name, bool async);
    virtual ~Logger() = default;

    template<typename... Args>
    void trace(std::string fmt, Args &&... args);

    template<typename... Args>
    void debug(std::string fmt, Args &&... args);

    template<typename... Args>
    void info(std::string fmt, Args &&... args);

    template<typename... Args>
    void warn(std::string fmt, Args &&... args);

    template<typename... Args>
    void error(std::string fmt, Args &&... args);

    template<typename... Args>
    void fatal(std::string fmt, Args &&... args);
  };
} // namespace torchserve
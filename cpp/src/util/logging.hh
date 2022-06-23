#ifndef CPP_UTIL_LOGGING_HH_
#define CPP_UTIL_LOGGING_HH_

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace torchserve {
  struct LoggerConfig {
    std::string logger_category,
    std::vector<std::string> sink_paths,
    std::string log_prefix_format,
    bool async
  };

  // TorchServe Logger API
  class Logger {
    public:
    Logger(const LoggerConfig& config);
    virtual ~Logger() = 0;

    virtual void Trace(const std::string& message) = 0;
    virtual void Debug(const std::string& message) = 0;
    virtual void Info(const std::string& message) = 0;
    virtual void Warn(const std::string& message) = 0;
    virtual void Error(const std::string& message) = 0;
    virtual void Fatal(const std::string& message) = 0;
  };

  // A Singleton class
  class LoggerStore {
    public:
    static LoggerStore& GetInstance();
    LoggerStore(const LoggerStore&) = delete;
    LoggerStore &operator=(const LoggerStore&) = delete;

    void RegisterLogger(std::shared_ptr<torchserve::Logger> new_logger);
    std::shared_ptr<torchserve::Logger> GetLogger(const std::string& logger_category);
    void drop(const std::string& logger_category);

    private:
    LoggerStore();
    ~LoggerStore();
    

    // key: logger_category
    // value: Logger
    std::map<std::string, std::shared_ptr<torchserve::Logger>> logger_table_;
    std::mutex logger_table_mutex_,
  };
} // namespace torchserve
#endif // CPP_UTIL_LOGGING_HH_
#include "src/utils/logging.hh"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace torchserve {

class TestLogger : public Logger {
    public:
    TestLogger(){};
    virtual ~TestLogger(){};
    void flush() { return Logger::logger->flush(); }
};

void Cleanup(const std::string& logfile_path) {
  if (!std::filesystem::remove(logfile_path)) {
    std::cout << "Failed to delete test log file" << logfile_path << std::endl;
  };
  torchserve::Logger::InitDefaultLogger();
}

TEST(LoggingTest, TestIncorrectLogInitialization) {
  std::string logger_config_path_str = "test/resources/logging/invalid.yaml";
  EXPECT_THROW(torchserve::Logger::InitLogger(logger_config_path_str),
               std::invalid_argument);
}


TEST(LoggingTest, TestFileLogInitialization) {
  std::string logfile_path = "test/resources/logging/test.log";
  std::string logger_config_path_str =
      "test/resources/logging/log_to_file.yaml";
  torchserve::Logger::InitLogger(logger_config_path_str);
  std::string log_line("Test");
  TS_LOG(INFO, log_line);
  EXPECT_TRUE(std::filesystem::exists(logfile_path));

  TestLogger test_logger;
  test_logger.flush();

  std::string contents;
  std::ifstream logfile(logfile_path);
  if (logfile.is_open()) {
    std::getline(logfile, contents);
    logfile.close();
  }
  EXPECT_TRUE(contents.compare(contents.size() - log_line.size(),
                               log_line.size(), log_line) == 0);

  // Cleanup
  Cleanup(logfile_path);
}
}  // namespace torchserve

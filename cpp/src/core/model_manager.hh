#pragma once

#include <deque>
#include <string>
#include <unordered_map>

#include "src/core/job.hh"
#include "src/core/model.hh"
#include "src/core/status.hh"

namespace torchserve {
class ModelManager {
  public:
  Status registerModel(const std::string &url, const std::string &defauleModelName, std::string &versionId);
  Status registerAndUpdateModel(const std::string &url, const std::string &defauleModelNam, std::string &versionId), ;
  Status unregistgerModel(const std::string &modelName, const std::string &versionId);
  Status addJob(Job &job);
  
  private:
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::unique_ptr<Model>>>
      mapModelNameVersion2Model;
};
}  // namespace torchserve
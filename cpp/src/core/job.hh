#pragma once

#include <string>
#include <vector>
#include "src/core/request.hh"
#include "src/core/response.hh"

namespace torchserve {
enum WorkerCommands {
  predict,
  load,
  unload,
  stats
};

// A job is a wrapper of an incoming request.
class Job {
  public:
  WorkerCommands jobCmd;
  std::string jobId;
  Request request;
  Response response;
  std::string modelName;  // one of the actor processing the request
  
  virtual void sendResponse() = 0;
  virtual void sendError(int status, std::string error) = 0;
};

// A RESTJob is the wrapper of an incoming request from HTTP.
class RESTJob : public Job {
 public:
  void sendResponse();
  void sendError(int status, std::string error);
};

// A GRPCJob is the wrapper of an incoming request from GRPC.
class GRPCJob : public Job {
 public:
  void sendResponse();
  void sendError(int status, std::string error);
};

class BatchJob {
  public:
  std::vector<std::shard_ptr<Job> > jobs;
  RequestBatch requestBatch;

  void addJob(std::shard_ptr<Job> &job);
};
}  // namespace torchserve
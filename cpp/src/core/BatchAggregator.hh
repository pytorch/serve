#pragma once

#include "job.hh"
#include "scheduler.hh"

namespace torchserve {
class BatchAggregator : public Scheduler {
 public:
  void enqueue(const std::unique_ptr<Job> &job);
};
}  // namespace torchserve
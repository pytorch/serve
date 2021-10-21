#pragma once

#include "blockingconcurrentqueue.h"
#include "src/core/job.hh"

namespace torchserve {

template <class T1>
class BatchAggregator : public Scheduler<T1> {
  public:
  void enqueue(const std::unique_ptr<T1> &job);
  
  private:
  // aggregate reads jobs from jobQueue, output BatchJob to next scheduler
  void aggregate();
  moodycamel::BlockingConcurrentQueue<Job> jobQueue;
};
}  // namespace torchserve

#pragma once

#include <memory>
#include "src/core/job.hh"
#include "src/core/request.hh"

namespace torchserve {

template <class T1>
class Scheduler {
  public:
  virtual void enqueue(const std::unique_ptr<T1> &task) = 0;

  template <class T2>
  void next(const std::unique_ptr<Scheduler<T2>> &scheduler);

  private:
  std::unique_ptr<Scheduler> next;
};
}  // namespace torchserve

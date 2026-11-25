// Persistent CPU workers for independent jet clustering histories
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "jet_rows.hh"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace geometry {

// One caller supplies consecutive batches to a worker group
class JetWorkers {
public:
    explicit JetWorkers(unsigned threads = std::thread::hardware_concurrency());
    ~JetWorkers();

    // Fill feature rows in source order and return the same status as jet_rows
    int run(int blocks, const double *mass_fit, const float *momenta, int events, int width, float *output);

private:
    void process();
    void wait_for_batch(std::uint64_t seen);

    const unsigned thread_limit;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready, done;
    std::uint64_t generation = 0;
    bool stopping = false;
    unsigned pending = 0;

    std::atomic<std::size_t> next{0};
    std::atomic<int> failure{0};
    int blocks = 0, width = 0, features = 0, chunk = 0, events = 0;
    const double *mass_fit = nullptr;
    const float *momenta = nullptr;
    float *output = nullptr;
};

} // namespace geometry

// Opaque worker group for language bindings
extern "C" void *jet_workers_create(unsigned threads);
extern "C" void jet_workers_destroy(void *workers);
extern "C" int jet_workers_run(void *workers, int blocks, const double *mass_fit, const float *momenta,
                               int events, int width, float *output);

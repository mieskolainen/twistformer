// Stream top-tagging observables from padded constituent four-momenta
//
// mikael.mieskolainen@cern.ch (2026)

#include "jet_workers.hh"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <cstdint>
#include <string>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// Read an unsigned decimal argument within the supported jet dimensions
int positive_integer(const char *arg) {
    const char *end = arg + std::strlen(arg);
    int value = 0;
    auto result = std::from_chars(arg, end, value);
    if (result.ec != std::errc{} || result.ptr != end || value <= 0)
        throw std::invalid_argument("Expected a positive integer");
    return value;
}

// Keep each input batch in source order while workers process independent jets
void extract(int width, int threads, int blocks) {
    constexpr std::array<double, 4> mass_fit{172.5, 80.4, 25., 15.};
    const int features = jet_feature_count(blocks, true);
    if (!features) throw std::invalid_argument("Unknown feature block mask");
    const std::size_t jet_size = 4 * static_cast<std::size_t>(width);
    const int capacity = static_cast<int>(std::min<std::size_t>(4096, std::max<std::size_t>(1, 4194304 / jet_size)));
    struct Batch {
        std::unique_ptr<float[]> input, output;
        int events = 0;
    };
    Batch batches[2];
    for (auto &batch : batches) {
        batch.input.reset(new float[jet_size * capacity]);
        batch.output.reset(new float[static_cast<std::size_t>(capacity) * features]);
    }
    auto read = [&](Batch &batch) {
        std::cin.read(reinterpret_cast<char *>(batch.input.get()), jet_size * capacity * sizeof(float));
        const auto bytes = static_cast<std::size_t>(std::cin.gcount());
        if (std::cin.bad() || (std::cin.fail() && !std::cin.eof())) throw std::runtime_error("Input read failed");
        if (bytes % (jet_size * sizeof(float))) throw std::runtime_error("Truncated input jet");
        batch.events = static_cast<int>(bytes / (jet_size * sizeof(float)));
    };
    auto write = [&](const Batch &batch) {
        std::cout.write(reinterpret_cast<const char *>(batch.output.get()),
                        static_cast<std::streamsize>(batch.events) * features * sizeof(float));
        if (!std::cout) throw std::runtime_error("Output write failed");
    };

    geometry::JetWorkers workers(threads);
    std::uint64_t complete = 0;
    int current = 0;
    read(batches[current]);
    while (batches[current].events) {
        auto &batch = batches[current];
        auto &other = batches[1 - current];
        // Stream the preceding output and next input while the current jets cluster
        auto io = std::async(std::launch::async, [&] {
            if (other.events) write(other);
            read(other);
        });
        const int status = workers.run(blocks, mass_fit.data(), batch.input.get(),
                                       batch.events, width, batch.output.get());
        io.get();
        if (status < 0) throw std::runtime_error("Invalid clustering arguments");
        if (status > 0) throw std::runtime_error("Invalid jet at input row " + std::to_string(complete + status));
        complete += batch.events;
        current = 1 - current;
    }
    if (complete) write(batches[1 - current]);
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("Output flush failed");
}
} // namespace

int main(int argc, char **argv) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    if (argc < 2 || argc > 4 || std::strcmp(argv[1], "--help") == 0) {
        std::cerr << "Usage: lund_rows WIDTH [THREADS [BLOCK_MASK]] < constituents.f32 > features.f32\n";
        return argc == 2 && std::strcmp(argv[1], "--help") == 0 ? 0 : 2;
    }
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        const int width = positive_integer(argv[1]);
        if (width > std::numeric_limits<int>::max()/2) throw std::invalid_argument("Constituent width too large");
        extract(width, argc >= 3 ? positive_integer(argv[2]) : std::max(1u, std::thread::hardware_concurrency()),
                argc == 4 ? positive_integer(argv[3]) : geometry::all);
    } catch (const std::exception &error) {
        std::cerr << "lund_rows: " << error.what() << '\n';
        return 1;
    }
}

// Batched observables from measured constituent four-momenta
//
// mikael.mieskolainen@cern.ch (2026)

#include "rest_hypergraph.hh"
#include "decay_partitions.hh"
#include "groomed_decay.hh"
#include "particle_histories.hh"
#include "color_flow.hh"
#include "simplex_rows.hh"
#include "gram_channels.hh"
#include "filtered_complex.hh"
#include "cascade_complex.hh"
#include "incidence_sheaf.hh"
#include "lab_references.hh"
#include "jet_workers.hh"

namespace g = geometry;

int jet_feature_count(int blocks, bool mass_fit) {
    constexpr int decay = g::rest | g::partitions | g::groomed | g::particle | g::color | g::simplex;
    if (blocks <= 0 || (blocks & ~(g::all | decay | g::gram | g::topology | g::cascade | g::sheaf | g::reference)) || ((blocks & decay) && !mass_fit)) return 0;
    const int history_width = mass_fit ? 266 : 246;
    return ((blocks & g::ca) ? history_width : 0) + ((blocks & g::kt) ? history_width : 0)
           + ((blocks & g::flow) ? 304 : 0) + ((blocks & g::rest) ? g::rest_graph_fields : 0)
           + ((blocks & g::partitions) ? 2*g::partition_fields : 0) + ((blocks & g::groomed) ? 3*g::groomed_fields : 0)
           + ((blocks & g::particle) ? g::particle_histogram_fields : 0) + ((blocks & g::color) ? 2*g::color_fields : 0)
           + ((blocks & g::simplex) ? g::simplices::output_fields : 0)
           + ((blocks & g::gram) ? g::gram_channels::fields : 0) + ((blocks & g::topology) ? g::complexes::output_fields : 0)
           + ((blocks & g::cascade) ? g::cascades::fields : 0) + ((blocks & g::sheaf) ? g::incidence::fields : 0)
           + ((blocks & g::reference) ? g::lab_references::fields : 0);
}

// Reuse the constituent buffer across one group of independent jets
static int compute_rows(int blocks, const double *mass_fit, const float *momenta, int events,
                        int width, float *output, int features) {
    g::JetList input;
    for (int event = 0; event < events; ++event) {
        try {
            input.clear();
            for (int i = 0; i < width; ++i) {
                const float *p = momenta + 4 * (static_cast<std::size_t>(event) * width + i);
                if (!std::all_of(p, p + 4, [](float x) { return std::isfinite(x); })) return event + 1;
                if (p[0] > 0 && (p[1] != 0 || p[2] != 0)) {
                    input.emplace_back(p[1], p[2], p[3], p[0]);
                    input.back().set_angles();
                }
            }
            float *row = output + static_cast<std::size_t>(features) * event;
            if (input.empty()) {
                std::fill_n(row, features, 0.f);
                continue;
            }
            float *reference_output = row + features - ((blocks & g::reference) ? g::lab_references::fields : 0);
            float *sheaf_output = reference_output - ((blocks & g::sheaf) ? g::incidence::fields : 0);
            float *cascade_output = sheaf_output - ((blocks & g::cascade) ? g::cascades::fields : 0);
            float *gram_output = cascade_output - ((blocks & g::gram) ? g::gram_channels::fields : 0);
            if (blocks & g::gram) {
                g::Momentum total;
                for (const auto &p : input) total = total + p;
                float *cursor = gram_output;
                g::gram_channels::project(input, total, cursor);
            }
            float *topology_output = gram_output - ((blocks & g::topology) ? g::complexes::output_fields : 0);
            float *simplex_output = topology_output - ((blocks & g::simplex) ? g::simplices::output_fields : 0);
            float *color_output = simplex_output - ((blocks & g::color) ? 2*g::color_fields : 0);
            float *particle_output = color_output - ((blocks & g::particle) ? g::particle_histogram_fields : 0);
            float *groomed_output = particle_output - ((blocks & g::groomed) ? 3*g::groomed_fields : 0);
            float *partition_output = groomed_output - ((blocks & g::partitions) ? 2*g::partition_fields : 0);
            if (blocks & (g::ca | g::groomed | g::cascade)) {
                const g::ClusterTree tree(input, false);
                if (blocks & g::cascade) g::cascades::project(tree, cascade_output);
                if (blocks & g::groomed) g::groomed_histories(tree, mass_fit, groomed_output);
                if (blocks & g::ca) g::clustering_history(input, tree, mass_fit, row);
            }
            if (blocks & (g::kt | g::flow | g::partitions | g::particle | g::color | g::simplex | g::topology | g::sheaf | g::reference)) {
                const g::ClusterTree tree(input, true);
                if (blocks & g::reference) g::lab_references::project(input, tree, reference_output);
                if (blocks & g::sheaf) g::incidence::project(input, tree, sheaf_output);
                if (blocks & g::topology) g::complexes::project(input, tree, topology_output);
                if (blocks & g::simplex) g::simplices::project(input, tree, mass_fit, simplex_output);
                if (blocks & g::color) g::color_flow(input, tree, mass_fit, color_output);
                if (blocks & g::particle) g::particle_histories(input, tree, mass_fit, particle_output);
                if (blocks & g::kt) g::clustering_history(input, tree, mass_fit, row);
                if (blocks & g::flow) g::angular_flow(input, tree, row);
                if (blocks & g::partitions) g::decay_partitions(input, tree, mass_fit, partition_output);
            }
            if (blocks & g::rest) g::rest_hypergraph(input, mass_fit, row);
        } catch (...) {
            return event + 1;
        }
    }
    return 0;
}

namespace geometry {

JetWorkers::JetWorkers(unsigned threads) : thread_limit(std::max(1u, threads)) {}

JetWorkers::~JetWorkers() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    ready.notify_all();
    for (auto &worker : workers) worker.join();
}

// Take small groups of jets to balance the work across different multiplicities
void JetWorkers::process() {
    for (std::size_t first = next.fetch_add(chunk); first < static_cast<std::size_t>(events); first = next.fetch_add(chunk)) {
        int result = compute_rows(blocks, mass_fit, momenta + static_cast<std::size_t>(first)*width*4,
                                  std::min(chunk, events - static_cast<int>(first)), width,
                                  output + static_cast<std::size_t>(first)*features, features);
        if (result) {
            const int row = static_cast<int>(first) + result;
            int previous = failure.load();
            while ((!previous || row < previous) && !failure.compare_exchange_weak(previous, row)) {}
        }
    }
}

void JetWorkers::wait_for_batch(std::uint64_t seen) {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
        ready.wait(lock, [&] { return stopping || generation != seen; });
        if (stopping) return;
        seen = generation;
        lock.unlock();
        process();
        lock.lock();
        if (--pending == 0) done.notify_one();
    }
}

int JetWorkers::run(int selected, const double *hypothesis, const float *input, int count, int slots, float *result) {
    const int columns = jet_feature_count(selected, hypothesis != nullptr);
    if (!columns || count < 0 || slots <= 0 || slots > std::numeric_limits<int>::max()/2
        || (count && (!input || !result))) return -1;
    if (hypothesis && (!std::all_of(hypothesis, hypothesis + 4, [](double x) { return std::isfinite(x); })
                       || hypothesis[2] <= 0. || hypothesis[3] <= 0.)) return -1;
    if (count == 0) return 0;
    const unsigned threads = std::min(thread_limit, static_cast<unsigned>(count));
    if (threads == 1) return compute_rows(selected, hypothesis, input, count, slots, result, columns);
    std::unique_lock<std::mutex> lock(mutex);
    while (workers.size() + 1 < threads)
        workers.emplace_back(&JetWorkers::wait_for_batch, this, generation);
    blocks = selected;
    mass_fit = hypothesis;
    momenta = input;
    events = count;
    width = slots;
    output = result;
    features = columns;
    chunk = std::clamp(count / static_cast<int>(threads) / 8, 1, 16);
    next = 0;
    failure = 0;
    pending = workers.size();
    ++generation;
    lock.unlock();
    ready.notify_all();
    process();
    lock.lock();
    done.wait(lock, [&] { return pending == 0; });
    return failure.load();
}

} // namespace geometry

int jet_rows(int blocks, const double *mass_fit, const float *momenta, int events, int width, float *output) {
    try {
        g::JetWorkers workers;
        return workers.run(blocks, mass_fit, momenta, events, width, output);
    } catch (...) {
        return -1;
    }
}

void *jet_workers_create(unsigned threads) {
    try {
        return new g::JetWorkers(threads);
    } catch (...) {
        return nullptr;
    }
}

void jet_workers_destroy(void *workers) {
    delete static_cast<g::JetWorkers *>(workers);
}

int jet_workers_run(void *workers, int blocks, const double *mass_fit, const float *momenta,
                    int events, int width, float *output) {
    if (!workers) return -1;
    try {
        return static_cast<g::JetWorkers *>(workers)->run(blocks, mass_fit, momenta, events, width, output);
    } catch (...) {
        return -1;
    }
}

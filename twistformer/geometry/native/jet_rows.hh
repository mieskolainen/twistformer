// Batched Lund and angular flow observables
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

namespace geometry {
enum Blocks { ca = 1, kt = 2, flow = 4, all = 7, rest = 8, partitions = 16, groomed = 32, particle = 64, color = 128, simplex = 256, topology = 512, gram = 1024, cascade = 2048, sheaf = 4096, reference = 8192 };
}

// Return the output width for the selected blocks and mass hypothesis
extern "C" int jet_feature_count(int blocks, bool mass_fit);

// Read [events, width, 4] energy-first float32 momenta and fill [events, features]
// mass_fit is null or four doubles in parent mass, pair mass, parent width, pair width order
// Return zero on success, a one-based failed jet index, or -1 for invalid arguments
extern "C" int jet_rows(int blocks, const double *mass_fit, const float *momenta,
                        int events, int width, float *output);

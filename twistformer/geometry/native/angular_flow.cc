// Multiscale radiation, exclusive subjet energy correlators and rest-frame moments
//
// mikael.mieskolainen@cern.ch (2026)

#include <algorithm>
#include <array>
#include <cmath>
#include "clustering.hh"
#include <vector>

namespace geometry {

// Contract radial flow moments and energy correlations through five microjets
void angular_flow(const JetList &input, const ClusterTree &seq, float *&out) {
    auto root = seq.exclusive_jets(1)[0];
    const double mass = std::max(root.m(), .001);
    double pt = 0.;
    for (auto &p : input)
        pt += p.pt();
    *out++ = std::abs(root.rap());
    *out++ = root.pt();
    *out++ = mass;
    *out++ = input.size();
    for (double cut : {.5, 1., 2., 4., 8., 16.}) {
        double count = 0., share = 0.;
        for (auto &p : input)
            if (p.pt() > cut) {
                ++count;
                share += p.pt() / pt;
            }
        *out++ = count;
        *out++ = share;
    }
    const int n = input.size();
    std::vector<std::array<double, 9>> neighbors(n);
    std::vector<std::array<double, 9>> counts(n);
    constexpr std::array<double, 9> radius = {.01, .025, .05, .075, .1, .15, .25, .4, .65};
    std::array<double, 14> spectrum{}, mixed{}, multiplicity{};
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double d = input[i].delta_R(input[j]);
            int bin = std::min(13, int(d / .05));
            spectrum[bin] += input[i].pt() * input[j].pt() / (pt * pt);
            mixed[bin] += (input[i].pt() + input[j].pt()) / pt;
            multiplicity[bin]++;
            auto start = std::upper_bound(radius.begin(), radius.end(), d) - radius.begin();
            if (start < 9) {
                neighbors[i][start] += input[j].pt() / pt;
                neighbors[j][start] += input[i].pt() / pt;
                counts[i][start]++;
                counts[j][start]++;
            }
        }
    for (int k = 0; k < 14; k++)
        for (double x : {spectrum[k], mixed[k], multiplicity[k]})
            *out++ = x;
    for (int k = 0; k < 9; k++) {
        double iso = 0., iso_pt = 0., occ = 0., maxshare = 0., sq = 0., cube = 0., cross = 0., sqcount = 0.;
        for (int i = 0; i < n; i++) {
            if (k) {
                counts[i][k] += counts[i][k - 1];
                neighbors[i][k] += neighbors[i][k - 1];
            }
            auto c = counts[i][k], z = input[i].pt() / pt, e = neighbors[i][k] + z;
            if (c == 0.) {
                ++iso;
                iso_pt += z;
            }
            occ += 1. / (c + 1.);
            maxshare = std::max(maxshare, e);
            sq += z * e * e;
            cube += z * e * e * e;
            cross += z * c;
            sqcount += z * c * c;
        }
        for (double x : {iso, iso_pt, occ, maxshare, sq, cube, cross, sqcount})
            *out++ = x;
    }
    auto axes = sorted_by_pt(seq.exclusive_jets(std::min(12, n)));
    int m = axes.size();
    double angles[12][12]{}, weights[12]{};
    for (int i = 0; i < m; i++) {
        weights[i] = axes[i].pt() / pt;
        for (int j = 0; j < i; j++)
            angles[i][j] = axes[i].delta_R(axes[j]);
    }
    for (double beta : {.5, 1., 2.}) {
        double e2 = 0., e3 = 0., e4 = 0., e5 = 0., min3 = 0., min4 = 0., min5 = 0., path4 = 0.;
        double v[12][12]{};
        for (int i = 0; i < m; i++)
            for (int j = 0; j < i; j++) {
                v[i][j] = std::pow(angles[i][j], beta);
                auto w2 = weights[i] * weights[j], a = v[i][j];
                e2 += w2 * a;
                for (int k = 0; k < j; k++) {
                    auto w3 = w2 * weights[k], b = v[i][k], c = v[j][k], tri = a * b * c;
                    auto ma = std::min({a, b, c});
                    min3 += w3 * ma;
                    e3 += w3 * tri;
                    for (int l = 0; l < k; l++) {
                        auto w4 = w3 * weights[l], d = v[i][l], e = v[j][l], f = v[k][l];
                        auto quad = tri * d * e * f, mb = std::min({ma, d, e, f});
                        min4 += w4 * mb;
                        e4 += w4 * quad;
                        path4 += w4 * mb * mb;
                        for (int h = 0; h < l; h++) {
                            auto w5 = w4 * weights[h], g = v[i][h], p = v[j][h], q = v[k][h], s = v[l][h];
                            e5 += w5 * quad * g * p * q * s;
                            min5 += w5 * std::min({mb, g, p, q, s});
                        }
                    }
                }
            }
        for (double x : {e2, e3, e4, e5, min3, min4, min5, path4, e3 / std::max(1.e-15, e2 * e2 * e2),
                         e4 * e2 / std::max(1.e-15, e3 * e3), e5 * e3 / std::max(1.e-15, e4 * e4),
                         min4 / std::max(1.e-14, min3), min5 / std::max(1.e-14, min4)})
            *out++ = x;
    }
    // Rest-frame Fox Wolfram moments from exclusive kT subjets
    for (int count : {4, 8, 16}) {
        if (root.m2() < 1.e-8) {
            for (int h = 0; h < 13; h++)
                *out++ = 0.;
            continue;
        }
        auto rest = seq.exclusive_jets(std::min(count, n));
        for (auto &p : rest)
            p.unboost(root);
        std::sort(rest.begin(), rest.end(), [](auto &a, auto &b) { return a.energy > b.energy; });
        double moments[11]{};
        for (size_t i = 0; i < rest.size(); i++)
            for (size_t j = 0; j <= i; j++) {
                auto &a = rest[i];
                auto &b = rest[j];
                double den = std::sqrt(a.modp2() * b.modp2());
                double c = std::clamp(
                    (a.x * b.x + a.y * b.y + a.z * b.z) / std::max(1.e-12, den), -1., 1.);
                double w = (i == j ? 1. : 2.) * (a.energy / mass) * (b.energy / mass);
                double prev = 1., curr = c;
                for (int h = 2; h <= 10; h++) {
                    double next = ((2 * h - 1) * c * curr - (h - 1) * prev) / h;
                    moments[h] += w * next;
                    prev = curr;
                    curr = next;
                }
            }
        for (int h = 2; h <= 10; h++)
            *out++ = moments[h];
        for (size_t i = 0; i < 4; i++)
            *out++ = i < rest.size() ? rest[i].energy / mass : 0.;
    }
    // Resolve the colour flow and the radial profile of each of the three hard cores
    auto prongs = sorted_by_pt(seq.exclusive_jets(std::min(3, n)));
    for (int i = 0; i < 3; i++) {
        std::array<double, 32> fields{};
        if (i < int(prongs.size())) {
            auto &p = prongs[i];
            double sum = 0.;
            auto parts = p.constituents();
            for (auto &q : parts)
                sum += q.pt();
            for (auto &q : parts) {
                double d = p.delta_R(q), z = q.pt() / sum;
                for (int k = 0; k < 6; k++)
                    if (d < radius[k + 1]) {
                        fields[k] += z;
                        fields[6 + k]++;
                    }
                for (int k = 0; k < 5; k++)
                    if (q.pt() > (1 << k)) {
                        fields[12 + k] += z;
                        fields[17 + k]++;
                    }
                fields[22] += std::sqrt(z);
                fields[23] += z * z;
                fields[24] += z * z * z;
                fields[25] += z * std::sqrt(d);
                fields[26] += z * d * d;
                fields[27] += z * d * (q.rap() - p.rap());
                fields[28] += z * d * p.delta_phi_to(q);
                fields[29] = std::max(fields[29], q.pt() / sum);
            }
            for (int j = 1; j <= 2; j++)
                if ((i + j) % 3 < int(prongs.size())) {
                    auto &b = prongs[(i + j) % 3];
                    fields[29 + j] = (fields[27] * (b.rap() - p.rap()) + fields[28] * p.delta_phi_to(b)) /
                                     std::max(.00001, p.delta_R(b));
                }
        }
        for (auto x : fields)
            *out++ = x;
    }
}

} // namespace geometry

// Copyright (c) 2026 Dr. Hyo Eun Ahn (ahn@aya.yale.edu)
// C++20 Deterministic 35x35 Moment-Space Tokamak Equilibrium & LMI Controller
#pragma once
#include <array>
#include <cstddef>

namespace kstar::tokamak {

constexpr std::size_t M_MOMENTS = 35;
constexpr std::size_t P_COILS = 4;

struct TokamakMomentOperator {
    std::array<std::array<double, M_MOMENTS>, M_MOMENTS> K{};
    std::array<double, M_MOMENTS> v0{};
    std::array<std::array<double, M_MOMENTS>, P_COILS> K_lmi{};
    std::array<double, M_MOMENTS> s_ref{};

    [[nodiscard]] constexpr std::array<double, M_MOMENTS>
    reconstruct_flux_coefficients(const std::array<double, M_MOMENTS>& s) const noexcept {
        std::array<double, M_MOMENTS> v = v0;
        for (std::size_t i = 0; i < M_MOMENTS; ++i) {
            double acc = v[i];
            for (std::size_t j = 0; j < M_MOMENTS; ++j) {
                acc += K[i][j] * s[j];
            }
            v[i] = acc;
        }
        return v;
    }

    [[nodiscard]] constexpr std::array<double, P_COILS>
    compute_lmi_coil_command(const std::array<double, M_MOMENTS>& s) const noexcept {
        std::array<double, P_COILS> u{};
        for (std::size_t p = 0; p < P_COILS; ++p) {
            double acc = 0.0;
            for (std::size_t j = 0; j < M_MOMENTS; ++j) {
                acc -= K_lmi[p][j] * (s[j] - s_ref[j]);
            }
            u[p] = acc;
        }
        return u;
    }
};

} // namespace kstar::tokamak

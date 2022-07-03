#ifndef LINEAR_WAVES_OPS_H
#define LINEAR_WAVES_OPS_H

#include "amr-wind/ocean_waves/regular_waves/LinearWaves.H"
#include "amr-wind/ocean_waves/OceanWavesOps.H"
#include "amr-wind/ocean_waves/OceanWaves.H"
#include "amr-wind/ocean_waves/regular_waves/regular_waves_ops.H"

namespace amr_wind {
namespace ocean_waves {

namespace ops {

template <>
struct ReadInputsOp<LinearWaves>
{
    void operator()(
        LinearWaves::DataType& data, const ::amr_wind::utils::MultiParser& pp)
    {
        auto& wdata = data.meta();
        auto& info = data.info();
        regular_waves::read_inputs(wdata, info, pp);
    }
};

template <>
struct InitDataOp<LinearWaves>
{
    void operator()(LinearWaves::DataType& data)
    {
        const auto& wdata = data.meta();

        auto& sim = data.sim();
        // cppcheck-suppress constVariable
        auto& levelset = sim.repo().get_field("levelset");
        auto nlevels = sim.repo().num_active_levels();
        auto geom = sim.mesh().Geom();
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& problo = geom[lev].ProbLoArray();
            const auto& dx = geom[lev].CellSizeArray();
            for (amrex::MFIter mfi(levelset(lev)); mfi.isValid(); ++mfi) {
                auto phi = levelset(lev).array(mfi);
                const amrex::Real depth = wdata.water_depth;
                const auto& bx = mfi.growntilebox();
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        phi(i, j, k) = z - depth;
                    });
            }
        }
    }
};

template <>
struct UpdateFSOp<LinearWaves>
{
    void operator()(LinearWaves::DataType& data)
    {
        const auto& wdata = data.meta();

        auto& sim = data.sim();
        // cppcheck-suppress constVariable
        auto& levelset = sim.repo().get_field("levelset");

        auto nlevels = sim.repo().num_active_levels();
        auto geom = sim.mesh().Geom();

        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& problo = geom[lev].ProbLoArray();
            const auto& dx = geom[lev].CellSizeArray();

            for (amrex::MFIter mfi(levelset(lev)); mfi.isValid(); ++mfi) {
                auto phi = levelset(lev).array(mfi);

                const amrex::Real height = wdata.wave_height;
                const amrex::Real wavelength = wdata.wave_length;
                const amrex::Real waterdepth = wdata.water_depth;

                const auto& bx = mfi.growntilebox();
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        const amrex::Real wavenumber = 2. * M_PI / wavelength;
                        const amrex::Real omega = std::pow(
                            wavenumber * 9.81 *
                                std::tanh(wavenumber * waterdepth),
                            0.5);
                        const amrex::Real phase = wavenumber * x - omega * time;
                        const amrex::Real eta = height / 2.0 * std::cos(phase);
                        phi(i, j, k) = z - eta;
                    });
            }
        }
    }
};

} // namespace ops
} // namespace ocean_waves
} // namespace amr_wind

#endif /* LINEAR_WAVES_OPS_H */

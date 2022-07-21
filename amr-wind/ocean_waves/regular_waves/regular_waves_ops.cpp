#include "amr-wind/ocean_waves/regular_waves/regular_waves_ops.H"
#include "amr-wind/physics/multiphase/MultiPhase.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/utilities/ncutils/nc_interface.H"
#include "amr-wind/utilities/io_utils.H"

#include "amr-wind/fvm/gradient.H"
#include "amr-wind/core/field_ops.H"

#include "amr-wind/ocean_waves/regular_waves/wave_utils_K.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace ocean_waves {
namespace regular_waves {

void read_inputs(
    RegularWavesBaseData& wdata,
    OceanWavesInfo& /*unused*/,
    const ::amr_wind::utils::MultiParser& pp)
{
    // Free surface zero level
    pp.query("zero_sea_level", wdata.zsl);
    pp.query("water_depth", wdata.water_depth);

    // Wave generation/absorption parameters
    pp.query("relax_zone_gen_length", wdata.gen_length);
    pp.query("numerical_beach_length", wdata.beach_length);
    pp.query("numerical_beach_start", wdata.x_start_beach);
}

void init_data_structures(RegularWavesBaseData& /*unused*/) {}

void apply_relaxation_zones(CFDSim& sim, RegularWavesBaseData& wdata)
{
    const int nlevels = sim.repo().num_active_levels();
    auto& m_ow_levelset = sim.repo().get_field("ow_levelset");
    auto& m_ow_vof = sim.repo().get_field("ow_vof");
    auto& m_ow_vel = sim.repo().get_field("ow_velocity");
    const auto& geom = sim.mesh().Geom();

    const auto& mphase = sim.physics_manager().get<MultiPhase>();
    const amrex::Real rho1 = mphase.rho1();
    const amrex::Real rho2 = mphase.rho2();

    for (int lev = 0; lev < nlevels; ++lev) {
        auto& ls = m_ow_levelset(lev);
        auto& target_vof = m_ow_vof(lev);
        const auto& dx = geom[lev].CellSizeArray();

        for (amrex::MFIter mfi(ls); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            const amrex::Array4<amrex::Real>& phi = ls.array(mfi);
            const amrex::Array4<amrex::Real>& volfrac = target_vof.array(mfi);
            const amrex::Real eps = 2. * std::cbrt(dx[0] * dx[1] * dx[2]);
            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    multiphase::levelset_to_vof(i, j, k, eps, phi, volfrac);
                });
        }
    }

    // Get time
    const auto& time = sim.time().new_time();
    amrex::Real ramp =
        (wdata.has_ramp) ? regular_waves::ramp(time, wdata.ramp_period) : 1.0;
    // Fill ghost and boundary cells before simulation begins
    m_ow_vof.fillpatch(time);

    auto& vof = sim.repo().get_field("vof");
    auto& velocity = sim.repo().get_field("velocity");
    auto& density = sim.repo().get_field("density");

    for (int lev = 0; lev < nlevels; ++lev) {
        for (amrex::MFIter mfi(vof(lev)); mfi.isValid(); ++mfi) {
            const auto& gbx = mfi.growntilebox(1);
            const auto& dx = geom[lev].CellSizeArray();
            const auto& problo = geom[lev].ProbLoArray();
            const auto& probhi = geom[lev].ProbHiArray();
            auto vel = velocity(lev).array(mfi);
            auto rho = density(lev).array(mfi);
            auto volfrac = vof(lev).array(mfi);
            auto target_volfrac = m_ow_vof(lev).array(mfi);
            auto target_vel = m_ow_vel(lev).array(mfi);

            const amrex::Real gen_length = wdata.gen_length;
            const amrex::Real x_start_beach = wdata.x_start_beach;
            const amrex::Real beach_length = wdata.beach_length;
            const amrex::Real zsl = wdata.zsl;

            amrex::ParallelFor(
                gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = amrex::min(
                        amrex::max(problo[0] + (i + 0.5) * dx[0], problo[0]),
                        probhi[0]);
                    const amrex::Real z = amrex::min(
                        amrex::max(problo[2] + (k + 0.5) * dx[2], problo[2]),
                        probhi[2]);

                    // Generation region
                    if (x <= gen_length) {
                        const amrex::Real Gamma =
                            regular_waves::Gamma_generate(x, gen_length);
                        const amrex::Real vf =
                            (1. - Gamma) * target_volfrac(i, j, k) * ramp +
                            Gamma * volfrac(i, j, k);
                        volfrac(i, j, k) = (vf > 1. - 1.e-10) ? 1.0 : vf;
                        vel(i, j, k, 0) =
                            (1. - Gamma) * target_vel(i, j, k, 0) *
                                volfrac(i, j, k) * ramp +
                            Gamma * vel(i, j, k, 0) * volfrac(i, j, k) +
                            (1. - volfrac(i, j, k)) * vel(i, j, k, 0);
                        vel(i, j, k, 1) =
                            (1. - Gamma) * target_vel(i, j, k, 1) *
                                volfrac(i, j, k) * ramp +
                            Gamma * vel(i, j, k, 1) * volfrac(i, j, k) +
                            (1. - volfrac(i, j, k)) * vel(i, j, k, 1);
                        vel(i, j, k, 2) =
                            (1. - Gamma) * target_vel(i, j, k, 2) *
                                volfrac(i, j, k) * ramp +
                            Gamma * vel(i, j, k, 2) * volfrac(i, j, k) +
                            (1. - volfrac(i, j, k)) * vel(i, j, k, 2);
                    }
                    // Numerical beach
                    if (x >= x_start_beach) {
                        const amrex::Real Gamma = regular_waves::Gamma_absorb(
                            x - x_start_beach, beach_length, 1.0);
                        volfrac(i, j, k) =
                            (1.0 - Gamma) * regular_waves::free_surface_to_vof(
                                                zsl, z, dx[2]) +
                            Gamma * volfrac(i, j, k);
                        vel(i, j, k, 0) =
                            Gamma * vel(i, j, k, 0) * volfrac(i, j, k);
                        vel(i, j, k, 1) =
                            Gamma * vel(i, j, k, 1) * volfrac(i, j, k);
                        vel(i, j, k, 2) =
                            Gamma * vel(i, j, k, 2) * volfrac(i, j, k);
                    }

                    // Make sure that density is updated before entering the
                    // solution
                    rho(i, j, k) = rho1 * volfrac(i, j, k) +
                                   rho2 * (1. - volfrac(i, j, k));
                });
        }
    }
}

void prepare_netcdf_file(
    const std::string& ncfile,
    const RegularWavesBaseData& meta,
    const OceanWavesInfo& info)
{
    amrex::ignore_unused(ncfile, meta, info);
}

void write_netcdf(
    const std::string& ncfile,
    const RegularWavesBaseData& meta,
    const OceanWavesInfo& info,
    const amrex::Real time)
{
    amrex::ignore_unused(ncfile, meta, info, time);
}

} // namespace regular_waves
} // namespace ocean_waves
} // namespace amr_wind
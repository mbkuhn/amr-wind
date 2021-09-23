#include "amr-wind/physics/multiphase/MultiPhase.H"
#include "amr-wind/physics/multiphase/numerical_wave_tank/NWT.H"
#include "amr-wind/physics/multiphase/numerical_wave_tank/wave_utils.H"
#include "amr-wind/utilities/trig_ops.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {

NWT::NWT(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.repo().get_field("velocity"))
    , m_levelset(sim.repo().get_field("levelset"))
    , m_vof(sim.repo().get_field("vof"))
{
    amrex::ParmParse pp(identifier());
    pp.query("amplitude", m_amplitude);
    pp.query("wavelength", m_wavelength);
    pp.query("zero_sea_level", m_zsl);
    pp.query("water_depth", m_waterdepth);

    // Wave generation/absorption parameters
    pp.query("relax_zone_gen_length", m_gen_length);
}

NWT::~NWT() = default;

/** Initialize the velocity and vof fields at the beginning of the
 *  simulation.
 */
void NWT::initialize_fields(int level, const amrex::Geometry& geom)
{

    auto& velocity = m_velocity(level);
    auto& levelset = m_levelset(level);
    velocity.setVal(0.0, 0, AMREX_SPACEDIM);

    const auto& dx = geom.CellSizeArray();
    const auto& problo = geom.ProbLoArray();
    const amrex::Real zsl = m_zsl;

    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.growntilebox();
        auto vel = velocity.array(mfi);
        auto phi = levelset.array(mfi);

        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                // const amrex::Real kappa = 2.0 * utils::pi() / lambda;
                // const amrex::Real epsilon = alpha * kappa;
                // Compute free surface amplitude
                // const amrex::Real eta = zsl;
                // +
                // alpha * ((1.0 - 1.0 / 16.0 * epsilon *
                // epsilon) *
                //             std::cos(kappa * x) +
                //         0.5 * epsilon * std::cos(2.0 * kappa
                //         * x) + 3.0 / 8.0 * epsilon * epsilon
                //         *
                //             std::cos(3.0 * kappa * x));

                phi(i, j, k) = zsl - z;
                vel(i, j, k, 0) = 0.0;
                vel(i, j, k, 1) = 0.0;
                vel(i, j, k, 2) = 0.0;
                // compute velocities
                // const amrex::Real g = 9.81;
                // const amrex::Real Omega =
                //    std::sqrt(g * kappa * (1.0 + epsilon * epsilon));
                // if (z < eta) {
                //    vel(i, j, k, 0) = Omega * alpha * std::exp(kappa * z) *
                //                      std::cos(kappa * x);
                //    vel(i, j, k, 2) = Omega * alpha * std::exp(kappa * z) *
                //                      std::sin(kappa * x);
                //}
            });
    }
}

void NWT::post_advance_work() { apply_relaxation_method(); }

void NWT::apply_relaxation_method()
{
    const auto& time = m_sim.time().current_time();
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    // Interpolate within the relazation zones
    for (int lev = 0; lev < nlevels; ++lev) {
        for (amrex::MFIter mfi(m_vof(lev)); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.growntilebox();
            const auto& dx = geom[lev].CellSizeArray();
            const auto& problo = geom[lev].ProbLoArray();
            auto vel = m_velocity(lev).array(mfi);
            auto volfrac = m_vof(lev).array(mfi);
            const amrex::Real wavelength = m_wavelength;
            const amrex::Real waterdepth = m_waterdepth;
            const amrex::Real amplitude = m_amplitude;

            const amrex::Real gen_length = m_gen_length;
            const amrex::Real ramp_period = m_ramp_period;
            const amrex::Real x_start_beach = m_x_start_beach;
            const amrex::Real absorb_length = m_absorb_length;
            const amrex::Real absorb_length_factor = m_absorb_length_factor;
            const amrex::Real grav = 9.81;
            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                    const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

                    const amrex::Real k_ = 2.0 * M_PI / wavelength;
                    const amrex::Real omega =
                        std::pow(k_ * grav * std::tanh(k_ * waterdepth), 0.5);
                    const amrex::Real phase = k_ * x - omega * time;

                    const amrex::Real eta = amplitude * std::sin(phase);
                    const amrex::Real u_w = omega * amplitude *
                                            std::cosh(k_ * (z + waterdepth)) /
                                            std::sinh(k_ * waterdepth) *
                                            std::cos(phase) * volfrac(i, j, k);
                    const amrex::Real w_w = omega * amplitude *
                                            std::sinh(k_ * (z + waterdepth)) /
                                            std::sinh(k_ * waterdepth) *
                                            std::sin(phase) * volfrac(i, j, k);
                    if (x < gen_length) {
                        const amrex::Real Gamma =
                            nwt::Gamma_generate(x, gen_length);
                        volfrac(i, j, k) =
                            (1.0 - Gamma) *
                                nwt::free_surface_to_vof(eta, z, dx[2]) *
                                nwt::ramp(time, ramp_period) +
                            Gamma * volfrac(i, j, k);
                        vel(i, j, k, 0) =
                            (1.0 - Gamma) * u_w * nwt::ramp(time, ramp_period) +
                            Gamma * vel(i, j, k, 0);
                        vel(i, j, k, 2) =
                            (1.0 - Gamma) * w_w * nwt::ramp(time, ramp_period) +
                            Gamma * vel(i, j, k, 2);
                    }
                    if (x > x_start_beach) {
                        const amrex::Real Gamma = nwt::Gamma_absorb(
                            x, absorb_length, absorb_length_factor);
                        volfrac(i, j, k) =
                            (1.0 - Gamma) *
                                nwt::free_surface_to_vof(eta, z, dx[2]) +
                            Gamma * volfrac(i, j, k);
                        vel(i, j, k, 0) = Gamma * vel(i, j, k, 0);
                        vel(i, j, k, 2) = Gamma * vel(i, j, k, 2);
                    }
                });
        }
    }
} // namespace amr_wind

} // namespace amr_wind
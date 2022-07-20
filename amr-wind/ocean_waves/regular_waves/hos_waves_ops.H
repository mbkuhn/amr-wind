#ifndef LINEAR_WAVES_OPS_H
#define LINEAR_WAVES_OPS_H

#include "amr-wind/ocean_waves/regular_waves/HOSWaves.H"
#include "amr-wind/ocean_waves/OceanWavesOps.H"
#include "amr-wind/ocean_waves/OceanWaves.H"
#include "amr-wind/ocean_waves/regular_waves/regular_waves_ops.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/physics/multiphase/MultiPhase.H"

namespace amr_wind {
namespace ocean_waves {

namespace ops {

template <>
struct ReadInputsOp<HOSWaves>
{
    void operator()(
        HOSWaves::DataType& data, const ::amr_wind::utils::MultiParser& pp)
    {
        auto& wdata = data.meta();
        auto& info = data.info();
        regular_waves::read_inputs(wdata, info, pp);

        pp.get("HOS_files_prefix", wdata.HOS_prefix);
        pp.query("HOS_init_timestep", wdata.HOS_n0);
        wdata.HOS_n = wdata.HOS_n0;
        pp.query("HOS_init_time",wdata.HOS_t);
        pp.get("HOS_dt",wdata.HOS_dt);
        // Routine assumes nodes of HOS are cells of amr-wind, requiring
        // - same resolution between HOS and amr-wind
        // - number of HOS nodes >= nx*ny of amr-wind
        // For when amr-wind sim is inset of HOS domain, offsets can be used
        pp.query("HOS_xind_offset", wdata.HOS_i0);
        pp.query("HOS_yind_offset", wdata.HOS_j0);
    }
};

template <>
struct InitDataOp<HOSWaves>
{
    void operator()(HOSWaves::DataType& data)
    {
        auto& wdata = data.meta();

        // Convert file string to stream
        std::stringstream fname;
        fname << wdata.HOS_prefix << "_" << wdata.HOS_n0 << ".dat";
        std::ifstream is (fname.str());
        // Read metadata from file
        std::string tmp;
        // Skip first two lines (TITLE and VARIABLES)
        std::getline(is, tmp);
        std::getline(is, tmp);
        // Skip "ZONE I="
        std::getline(is, tmp,'=');
        // Get dimension i
        std::getline(is, tmp,',');
        wdata.HOS_nx = std::stoi(tmp);
        // Skip "J ="
        std::getline(is, tmp,'=');
        // Get dimension j
        std::getline(is, tmp);
        wdata.HOS_ny = std::stoi(tmp);
        // Check if current dimensions are compatible

        // Allocate arrays for storage
        wdata.eta.resize(wdata.HOS_nx * wdata.HOS_ny);
        wdata.u.resize(wdata.HOS_nx * wdata.HOS_ny);
        wdata.v.resize(wdata.HOS_nx * wdata.HOS_ny);
        wdata.w.resize(wdata.HOS_nx * wdata.HOS_ny);
        // Skip next line to get to data
        std::getline(is, tmp);
        // Read interface heights and velocities
        amrex::Real buf; // to skip entries
        for (int ipt = 0; ipt < wdata.HOS_nx * wdata.HOS_ny; ++ ipt) {
            // Skip x, y and save eta, u, v, w
            is >> buf >> buf >> wdata.eta[ipt] >> wdata.u[ipt] >>
                wdata.v[ipt] >> wdata.w[ipt];
        }

        auto& sim = data.sim();
        // cppcheck-suppress constVariable
        // Record phase densities
        const auto& mphase = sim.physics_manager().get<MultiPhase>();
        const amrex::Real rho1 = mphase.rho1();
        const amrex::Real rho2 = mphase.rho2();
        // Save current time
        wdata.t_last = sim.time().current_time();
        auto& m_ow_levelset = sim.repo().get_field("ow_levelset");
        auto& m_ow_velocity = sim.repo().get_field("ow_velocity");
        auto& velocity = sim.repo().get_field("velocity");
        auto& vof = sim.repo().get_field("vof");
        auto& density = sim.repo().get_field("density");
        auto nlevels = sim.repo().num_active_levels();
        auto geom = sim.mesh().Geom();
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& problo = geom[lev].ProbLoArray();
            const auto& dx = geom[lev].CellSizeArray();
            for (amrex::MFIter mfi(m_ow_levelset(lev)); mfi.isValid(); ++mfi) {
                auto phi = m_ow_levelset(lev).array(mfi);
                auto vel = velocity(lev).array(mfi);
                auto ow_vel = m_ow_velocity(lev).array(mfi);
                auto volfrac = vof(lev).array(mfi);
                auto rho = density(lev).array(mfi);
                const amrex::Real zsl = wdata.zsl;
                const int i0 = wdata.HOS_i0;
                const int j0 = wdata.HOS_j0;
                const int hos_nx = wdata.HOS_nx;
                const auto& vbx = mfi.validbox();
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        phi(i, j, k) =
                            wdata.eta[i + i0 + (j + j0) * hos_nx] + zsl - z;
                        amrex::Real efac = 0.0;
                        if (phi(i, j, k) > 0.0) {
                            efac = exp(-phi(i, j, k));
                        } else {
                            if (phi(i, j, k) > -0.5 * dx[2]) {
                                efac = 0.5 *
                                       (1.0 + exp(0.5 * dx[2] - phi(i, j, k)));
                            }
                        }
                        vel(i, j, k, 0) =
                            wdata.u[i + i0 + (j + j0) * hos_nx] * efac;
                        vel(i, j, k, 1) =
                            wdata.v[i + i0 + (j + j0) * hos_nx] * efac;
                        vel(i, j, k, 2) =
                            wdata.w[i + i0 + (j + j0) * hos_nx] * efac;
                        // ow velocity should initially be in sync with velocity
                        ow_vel(i, j, k, 0) = vel(i, j, k, 0);
                        ow_vel(i, j, k, 1) = vel(i, j, k, 1);
                        ow_vel(i, j, k, 2) = vel(i, j, k, 2);
                    });
                // Translate levelset to vof
                const amrex::Real eps = 2. * std::cbrt(dx[0] * dx[1] * dx[2]);
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        multiphase::levelset_to_vof(i, j, k, eps, phi, volfrac);
                        rho(i, j, k) =
                            rho1 * volfrac(i, j, k) + rho2 * (1.0 - volfrac(i, j, k));
                    });
            }
        }
        vof.fillpatch(0.0);
        density.fillpatch(0.0);
    }
};

template <>
struct UpdateRelaxZonesOp<HOSWaves>
{
    void operator()(HOSWaves::DataType& data)
    {
        auto& wdata = data.meta();

        auto& sim = data.sim();
        // cppcheck-suppress constVariable
        // Get time at conclusion of current timestep
        const auto& time = sim.time().new_time();
        // Check if HOS time needs to be incremented
        if (time > wdata.HOS_t) {
            // Increment time step
            ++wdata.HOS_n;
            // Increment time
            wdata.HOS_t += wdata.HOS_dt;

            // Convert file string to stream
            std::stringstream fname;
            fname << wdata.HOS_prefix << "_" << wdata.HOS_n << ".dat";
            std::ifstream is(fname.str());
            // Read metadata from file
            std::string tmp;
            // Skip title, variables, dimensions, line after
            std::getline(is, tmp);
            std::getline(is, tmp);
            std::getline(is, tmp);
            std::getline(is, tmp);
            // Read interface heights and velocities
            amrex::Real buf; // to skip entries
            for (int ipt = 0; ipt < wdata.HOS_nx * wdata.HOS_ny; ++ipt) {
                // Skip x, y and save eta, u, v, w
                is >> buf >> buf >> wdata.eta[ipt] >> wdata.u[ipt] >>
                    wdata.v[ipt] >> wdata.w[ipt];
            }
        }

        // Update ow fields every time
        auto& m_ow_levelset = sim.repo().get_field("ow_levelset");
        auto& m_ow_velocity = sim.repo().get_field("ow_velocity");

        auto nlevels = sim.repo().num_active_levels();
        auto geom = sim.mesh().Geom();
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& problo = geom[lev].ProbLoArray();
            const auto& dx = geom[lev].CellSizeArray();
            for (amrex::MFIter mfi(m_ow_levelset(lev)); mfi.isValid(); ++mfi) {
                auto phi = m_ow_levelset(lev).array(mfi);
                auto vel = m_ow_velocity(lev).array(mfi);

                const amrex::Real zsl = wdata.zsl;
                const int i0 = wdata.HOS_i0;
                const int j0 = wdata.HOS_j0;
                const int hos_nx = wdata.HOS_nx;
                const auto& vbx = mfi.validbox();
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        // Interpolate temporally
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        phi(i, j, k) +=
                            ((wdata.eta[i + i0 + (j + j0) * hos_nx] + zsl - z) -
                             phi(i, j, k)) /
                            (wdata.HOS_t - wdata.t_last) * (time - wdata.t_last);
                        amrex::Real efac = 0.0;
                        if (phi(i, j, k) > 0.0) {
                            efac = exp(-phi(i, j, k));
                        } else {
                            if (phi(i, j, k) > -0.5 * dx[2]) {
                                efac = 0.5 *
                                       (1.0 + exp(0.5 * dx[2] - phi(i, j, k)));
                            }
                        }
                        vel(i, j, k, 0) +=
                            ((wdata.u[i + i0 + (j + j0) * hos_nx] * efac) -
                             vel(i, j, k, 0)) /
                            (wdata.HOS_t - wdata.t_last) * (time - wdata.t_last);
                        vel(i, j, k, 1) +=
                            ((wdata.v[i + i0 + (j + j0) * hos_nx] * efac) -
                             vel(i, j, k, 1)) /
                            (wdata.HOS_t - wdata.t_last) * (time - wdata.t_last);
                        vel(i, j, k, 2) +=
                            ((wdata.w[i + i0 + (j + j0) * hos_nx] * efac) -
                             vel(i, j, k, 2)) /
                            (wdata.HOS_t - wdata.t_last) * (time - wdata.t_last);
                    });
            }
        }
        m_ow_levelset.fillpatch(0.0);
        // Record current time as 'last' for next timestep
        wdata.t_last = time;
    }
};

} // namespace ops
} // namespace ocean_waves
} // namespace amr_wind

#endif /* LINEAR_WAVES_OPS_H */

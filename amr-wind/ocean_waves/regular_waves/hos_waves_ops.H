#ifndef HOS_WAVES_OPS_H
#define HOS_WAVES_OPS_H

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
        fname << wdata.HOS_prefix << "_" << wdata.HOS_n0 << ".txt";
        std::ifstream is (fname.str());
        // Read metadata from file
        std::string tmp;
        // Get initial time
        std::getline(is,tmp,'=');
        std::getline(is,tmp);
        wdata.HOS_t = std::stof(tmp);
        // Get dt
        std::getline(is,tmp,'=');
        std::getline(is,tmp);
        wdata.HOS_dt = std::stof(tmp);
        // Get nx, Lx
        std::getline(is, tmp,'=');
        std::getline(is, tmp,',');
        wdata.HOS_nx = std::stoi(tmp);
        std::getline(is, tmp,'=');
        std::getline(is, tmp);
        wdata.HOS_Lx = std::stof(tmp);
        // Get ny, Ly
        std::getline(is, tmp,'=');
        std::getline(is, tmp,',');
        wdata.HOS_ny = std::stoi(tmp);
        std::getline(is, tmp,'=');
        std::getline(is, tmp);
        wdata.HOS_Ly = std::stof(tmp);
        // Get nz, zmin, zmax
        std::getline(is, tmp,'=');
        std::getline(is, tmp,',');
        wdata.HOS_nz = std::stoi(tmp);
        std::getline(is, tmp,'=');
        std::getline(is, tmp,',');
        wdata.HOS_zmin = std::stof(tmp);
        std::getline(is, tmp,'=');
        std::getline(is, tmp);
        wdata.HOS_zmax = std::stof(tmp);

        // Modify Lx, Ly to account for node - cell difference
        wdata.HOS_Lx *=
            (((amrex::Real)wdata.HOS_nx) / ((amrex::Real)wdata.HOS_nx - 1));
        wdata.HOS_Ly *=
            (((amrex::Real)wdata.HOS_ny) / ((amrex::Real)wdata.HOS_ny - 1));

        // Allocate arrays for storage
        wdata.eta.resize(wdata.HOS_nx * wdata.HOS_ny);
        wdata.u.resize(wdata.HOS_nx * wdata.HOS_ny * wdata.HOS_nz);
        wdata.v.resize(wdata.HOS_nx * wdata.HOS_ny * wdata.HOS_nz);
        wdata.w.resize(wdata.HOS_nx * wdata.HOS_ny * wdata.HOS_nz);
        // Skip key
        std::getline(is,tmp);
        // Read interface heights and velocities
        for (int ilat = 0; ilat < wdata.HOS_nx * wdata.HOS_ny; ++ilat) {
            // Get eta for current point
            is >> wdata.eta[ilat];
            // Get u, v, w for full depth of 2D point
            for (int ivert = 0; ivert < wdata.HOS_nz; ++ivert) {
                is >> wdata.u[ilat * wdata.HOS_nz + ivert] >>
                    wdata.v[ilat * wdata.HOS_nz + ivert] >>
                    wdata.w[ilat * wdata.HOS_nz + ivert];
            }
        }

        auto& sim = data.sim();
        // cppcheck-suppress constVariable
        // Check if current dimensions are compatible
        if ((sim.mesh().Geom()[0]).ProbLoArray()[0] < -1e-6 ||
            (sim.mesh().Geom()[0]).ProbHiArray()[0] >
                wdata.HOS_Lx * (1.0 + 1e-6) ||
            (sim.mesh().Geom()[0]).ProbLoArray()[1] < -1e-6 ||
            (sim.mesh().Geom()[0]).ProbHiArray()[1] >
                wdata.HOS_Ly * (1.0 + 1e-6)) {
            amrex::Abort("HOS OceanWaves: Lateral dimensions incompatible.");
        }

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
                const amrex::Real zmin = wdata.HOS_zmin;
                const amrex::Real zmax = wdata.HOS_zmax;
                const int hos_nx = wdata.HOS_nx;
                const int hos_ny = wdata.HOS_ny;
                const int hos_nz = wdata.HOS_nz;
                const amrex::Real hos_dx = wdata.HOS_Lx / hos_nx;
                const amrex::Real hos_dy = wdata.HOS_Ly / hos_ny;
                const amrex::Real hos_dz = (zmax-zmin) / hos_nz;
                const auto& vbx = mfi.validbox();
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                        const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        // Determine lateral indices for interpolation
                        // It's best to have the grids line up directly
                        int ii = 0;
                        amrex::Real HOS_x = hos_dx * (ii + 0.5);
                        while (HOS_x < x) {
                            ++ii;
                            HOS_x = hos_dx * (ii + 0.5);
                        }
                        int jj = 0;
                        amrex::Real HOS_y = hos_dy * (jj + 0.5);
                        while (HOS_y < y) {
                            ++jj;
                            HOS_y = hos_dy * (jj + 0.5);
                        }

                        if (z > zmax || z < zmin) {
                            phi(i, j, k) =
                                wdata.eta[ii + jj * hos_nx] + zsl - z;
                            // Assume velocities are zero outside range
                            vel(i, j, k, 0) = 0;
                            vel(i, j, k, 1) = 0;
                            vel(i, j, k, 2) = 0;
                        } else {
                            // Determine vertical indices
                            int kk = 0;
                            amrex::Real HOS_z =
                                zmin + hos_dz * (kk + 0.5);
                            while (HOS_z < z) {
                                ++kk;
                                HOS_z = zmin + hos_dz * (kk + 0.5);
                            }
                            // Shift indices left
                            ii = ii - 1;
                            jj = jj - 1;
                            kk = kk - 1;
                            // Modify indices to avoid going out of bounds
                            ii = amrex::max(0,amrex::min(hos_nx-1,ii));
                            jj = amrex::max(0,amrex::min(hos_ny-1,jj));
                            kk = amrex::max(0,amrex::min(hos_nz-1,kk));
                            // Reset locations
                            HOS_x = hos_dx * (ii + 0.5);
                            HOS_y = hos_dy * (jj + 0.5);
                            HOS_z = zmin + hos_dz * (kk + 0.5);
                            // Interpolation factors
                            amrex::Real wx_hi = (x - HOS_x) / hos_dx;
                            amrex::Real wy_hi = (y - HOS_y) / hos_dy;
                            amrex::Real wz_hi = (z - HOS_z) / hos_dz;
                            amrex::Real wx_lo = 1.0 - wx_hi;
                            amrex::Real wy_lo = 1.0 - wy_hi;
                            amrex::Real wz_lo = 1.0 - wz_hi;

                            // Get data
                            const amrex::Real eta_00 =
                                wdata.eta[ii + jj * hos_nx];
                            const amrex::Real eta_01 =
                                wdata.eta[ii + (jj + 1) * hos_nx];
                            const amrex::Real eta_10 =
                                wdata.eta[ii + 1 + jj * hos_nx];
                            const amrex::Real eta_11 =
                                wdata.eta[ii + 1 + (jj + 1) * hos_nx];
                            const amrex::Real eta_ = wx_lo * wy_lo * eta_00 +
                                                     wx_lo * wy_hi * eta_01 +
                                                     wx_hi * wy_lo * eta_10 +
                                                     wx_hi * wy_hi * eta_11;

                            phi(i, j, k) = eta_ + zsl - z;

                            // Set up indices for clarity
                            const int i000 = (ii + jj * hos_nx) * hos_nz + kk;
                            const int i001 =
                                (ii + jj * hos_nx) * hos_nz + kk + 1;
                            const int i010 =
                                (ii + (jj + 1) * hos_nx) * hos_nz + kk;
                            const int i100 =
                                (ii + 1 + jj * hos_nx) * hos_nz + kk;
                            const int i011 =
                                (ii + (jj + 1) * hos_nx) * hos_nz + kk + 1;
                            const int i101 =
                                (ii + 1 + jj * hos_nx) * hos_nz + kk + 1;
                            const int i110 =
                                (ii + 1 + (jj + 1) * hos_nx) * hos_nz + kk;
                            const int i111 =
                                (ii + 1 + (jj + 1) * hos_nx) * hos_nz + kk + 1;
                            vel(i, j, k, 0) =
                                wx_lo * wy_lo * wz_lo * wdata.u[i000] +
                                wx_lo * wy_lo * wz_hi * wdata.u[i001] +
                                wx_lo * wy_hi * wz_lo * wdata.u[i010] +
                                wx_lo * wy_hi * wz_hi * wdata.u[i011] +
                                wx_hi * wy_lo * wz_lo * wdata.u[i100] +
                                wx_hi * wy_lo * wz_hi * wdata.u[i101] +
                                wx_hi * wy_hi * wz_lo * wdata.u[i110] +
                                wx_hi * wy_hi * wz_hi * wdata.u[i111];
                            vel(i, j, k, 1) =
                                wx_lo * wy_lo * wz_lo * wdata.v[i000] +
                                wx_lo * wy_lo * wz_hi * wdata.v[i001] +
                                wx_lo * wy_hi * wz_lo * wdata.v[i010] +
                                wx_lo * wy_hi * wz_hi * wdata.v[i011] +
                                wx_hi * wy_lo * wz_lo * wdata.v[i100] +
                                wx_hi * wy_lo * wz_hi * wdata.v[i101] +
                                wx_hi * wy_hi * wz_lo * wdata.v[i110] +
                                wx_hi * wy_hi * wz_hi * wdata.v[i111];
                            vel(i, j, k, 2) =
                                wx_lo * wy_lo * wz_lo * wdata.w[i000] +
                                wx_lo * wy_lo * wz_hi * wdata.w[i001] +
                                wx_lo * wy_hi * wz_lo * wdata.w[i010] +
                                wx_lo * wy_hi * wz_hi * wdata.w[i011] +
                                wx_hi * wy_lo * wz_lo * wdata.w[i100] +
                                wx_hi * wy_lo * wz_hi * wdata.w[i101] +
                                wx_hi * wy_hi * wz_lo * wdata.w[i110] +
                                wx_hi * wy_hi * wz_hi * wdata.w[i111];
                        }
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

            std::stringstream fname;
            fname << wdata.HOS_prefix << "_" << wdata.HOS_n << ".txt";
            std::ifstream is (fname.str());
            // Skip metadata
            std::string tmp;
            std::getline(is,tmp); // Time
            std::getline(is,tmp); // Timestep
            std::getline(is,tmp); // x dimensions
            std::getline(is,tmp); // y dimensions
            std::getline(is,tmp); // z dimensions
            std::getline(is,tmp); // key

            // Read interface heights and velocities
            for (int ilat = 0; ilat < wdata.HOS_nx * wdata.HOS_ny; ++ilat) {
                // Get eta for current point
                is >> wdata.eta[ilat];
                // Get u, v, w for full depth of 2D point
                for (int ivert = 0; ivert < wdata.HOS_nz; ++ivert) {
                    is >> wdata.u[ilat * wdata.HOS_nz + ivert] >>
                        wdata.v[ilat * wdata.HOS_nz + ivert] >>
                        wdata.w[ilat * wdata.HOS_nz + ivert];
                }
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
                const amrex::Real zmin = wdata.HOS_zmin;
                const amrex::Real zmax = wdata.HOS_zmax;
                const int hos_nx = wdata.HOS_nx;
                const int hos_ny = wdata.HOS_ny;
                const int hos_nz = wdata.HOS_nz;
                const amrex::Real hos_dx = wdata.HOS_Lx / hos_nx;
                const amrex::Real hos_dy = wdata.HOS_Ly / hos_ny;
                const amrex::Real hos_dz = (zmax-zmin) / hos_nz;
                const auto& vbx = mfi.validbox();
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                        const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        // Determine lateral indices for interpolation
                        // It's best to have the grids line up directly
                        int ii = 0;
                        amrex::Real HOS_x = hos_dx * (ii + 0.5);
                        while (HOS_x < x) {
                            ++ii;
                            HOS_x = hos_dx * (ii + 0.5);
                        }
                        int jj = 0;
                        amrex::Real HOS_y = hos_dy * (jj + 0.5);
                        while (HOS_y < y) {
                            ++jj;
                            HOS_y = hos_dy * (jj + 0.5);
                        }

                        // Values at HOS_t
                        amrex::Real HOS_phi, HOS_u, HOS_v, HOS_w;

                        if (z > zmax || z < zmin) {
                            HOS_phi =
                                wdata.eta[ii + jj * hos_nx] + zsl - z;
                            // Assume velocities are zero outside range
                            HOS_u = 0;
                            HOS_v = 0;
                            HOS_w = 0;
                        } else {
                            // Determine vertical indices
                            int kk = 0;
                            amrex::Real HOS_z = zmin + hos_dz * (kk + 0.5);
                            while (HOS_z < z) {
                                ++kk;
                                HOS_z = zmin + hos_dz * (kk + 0.5);
                            }
                            // Shift indices left
                            ii = ii - 1;
                            jj = jj - 1;
                            kk = kk - 1;
                            // Modify indices to avoid going out of bounds
                            ii = amrex::max(0, amrex::min(hos_nx - 1, ii));
                            jj = amrex::max(0, amrex::min(hos_ny - 1, jj));
                            kk = amrex::max(0, amrex::min(hos_nz - 1, kk));
                            // Reset locations
                            HOS_x = hos_dx * (ii + 0.5);
                            HOS_y = hos_dy * (jj + 0.5);
                            HOS_z = zmin + hos_dz * (kk + 0.5);
                            // Interpolation factors
                            amrex::Real wx_hi = (x - HOS_x) / hos_dx;
                            amrex::Real wy_hi = (y - HOS_y) / hos_dy;
                            amrex::Real wz_hi = (z - HOS_z) / hos_dz;
                            amrex::Real wx_lo = 1.0 - wx_hi;
                            amrex::Real wy_lo = 1.0 - wy_hi;
                            amrex::Real wz_lo = 1.0 - wz_hi;

                            // Get data
                            const amrex::Real eta_00 =
                                wdata.eta[ii + jj * hos_nx];
                            const amrex::Real eta_01 =
                                wdata.eta[ii + (jj + 1) * hos_nx];
                            const amrex::Real eta_10 =
                                wdata.eta[ii + 1 + jj * hos_nx];
                            const amrex::Real eta_11 =
                                wdata.eta[ii + 1 + (jj + 1) * hos_nx];
                            const amrex::Real eta_ = wx_lo * wy_lo * eta_00 +
                                                     wx_lo * wy_hi * eta_01 +
                                                     wx_hi * wy_lo * eta_10 +
                                                     wx_hi * wy_hi * eta_11;

                            HOS_phi = eta_ + zsl - z;

                            // Set up indices for clarity
                            const int i000 = (ii + jj * hos_nx) * hos_nz + kk;
                            const int i001 =
                                (ii + jj * hos_nx) * hos_nz + kk + 1;
                            const int i010 =
                                (ii + (jj + 1) * hos_nx) * hos_nz + kk;
                            const int i100 =
                                (ii + 1 + jj * hos_nx) * hos_nz + kk;
                            const int i011 =
                                (ii + (jj + 1) * hos_nx) * hos_nz + kk + 1;
                            const int i101 =
                                (ii + 1 + jj * hos_nx) * hos_nz + kk + 1;
                            const int i110 =
                                (ii + 1 + (jj + 1) * hos_nx) * hos_nz + kk;
                            const int i111 =
                                (ii + 1 + (jj + 1) * hos_nx) * hos_nz + kk + 1;
                            HOS_u =
                                wx_lo * wy_lo * wz_lo * wdata.u[i000] +
                                wx_lo * wy_lo * wz_hi * wdata.u[i001] +
                                wx_lo * wy_hi * wz_lo * wdata.u[i010] +
                                wx_lo * wy_hi * wz_hi * wdata.u[i011] +
                                wx_hi * wy_lo * wz_lo * wdata.u[i100] +
                                wx_hi * wy_lo * wz_hi * wdata.u[i101] +
                                wx_hi * wy_hi * wz_lo * wdata.u[i110] +
                                wx_hi * wy_hi * wz_hi * wdata.u[i111];
                            HOS_v =
                                wx_lo * wy_lo * wz_lo * wdata.v[i000] +
                                wx_lo * wy_lo * wz_hi * wdata.v[i001] +
                                wx_lo * wy_hi * wz_lo * wdata.v[i010] +
                                wx_lo * wy_hi * wz_hi * wdata.v[i011] +
                                wx_hi * wy_lo * wz_lo * wdata.v[i100] +
                                wx_hi * wy_lo * wz_hi * wdata.v[i101] +
                                wx_hi * wy_hi * wz_lo * wdata.v[i110] +
                                wx_hi * wy_hi * wz_hi * wdata.v[i111];
                            HOS_w =
                                wx_lo * wy_lo * wz_lo * wdata.w[i000] +
                                wx_lo * wy_lo * wz_hi * wdata.w[i001] +
                                wx_lo * wy_hi * wz_lo * wdata.w[i010] +
                                wx_lo * wy_hi * wz_hi * wdata.w[i011] +
                                wx_hi * wy_lo * wz_lo * wdata.w[i100] +
                                wx_hi * wy_lo * wz_hi * wdata.w[i101] +
                                wx_hi * wy_hi * wz_lo * wdata.w[i110] +
                                wx_hi * wy_hi * wz_hi * wdata.w[i111];
                        }
                        // Interpolate temporally
                        phi(i, j, k) += (HOS_phi - phi(i, j, k)) /
                                        (wdata.HOS_t - wdata.t_last) *
                                        (time - wdata.t_last);
                        vel(i, j, k, 0) += (HOS_u - vel(i, j, k, 0)) /
                                           (wdata.HOS_t - wdata.t_last) *
                                           (time - wdata.t_last);
                        vel(i, j, k, 1) += (HOS_v - vel(i, j, k, 1)) /
                                           (wdata.HOS_t - wdata.t_last) *
                                           (time - wdata.t_last);
                        vel(i, j, k, 2) += (HOS_w - vel(i, j, k, 2)) /
                                           (wdata.HOS_t - wdata.t_last) *
                                           (time - wdata.t_last);
                    });
            }
        }
        m_ow_levelset.fillpatch(0.0);
        m_ow_velocity.fillpatch(0.0);
        // Record current time as 'last' for next timestep
        wdata.t_last = time;
    }
};

} // namespace ops
} // namespace ocean_waves
} // namespace amr_wind

#endif /* LINEAR_WAVES_OPS_H */

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
        // Read interface heights and velocities
        for (int ilat = 0; ilat < wdata.HOS_nx * wdata.HOS_ny; ++ilat) {
            // Get eta for current point
            std::getline(is, tmp, '=');
            is >> wdata.eta[ilat];
            // Get u, v, w for full depth of 2D point
            for (int ivert = 0; ivert < wdata.HOS_nz; ++ivert) {
                std::getline(is, tmp, '=');
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

                            vel(i, j, k, 0) =
                                wx_lo * wy_lo * wz_lo *
                                    wdata.u[(ii + jj * hos_nx) * hos_nz + kk] +
                                wx_lo * wy_lo * wz_hi *
                                    wdata
                                        .u[(ii + jj * hos_nx) * hos_nz + kk +
                                           1] +
                                wx_lo * wy_hi * wz_lo *
                                    wdata
                                        .u[(ii + (jj + 1) * hos_nx) * hos_nz +
                                           kk] +
                                wx_lo * wy_hi * wz_hi *
                                    wdata
                                        .u[(ii + (jj + 1) * hos_nx) * hos_nz +
                                           kk + 1] +
                                wx_hi * wy_lo * wz_lo *
                                    wdata
                                        .u[(ii + 1 + jj * hos_nx) * hos_nz +
                                           kk] +
                                wx_hi * wy_lo * wz_hi *
                                    wdata
                                        .u[(ii + 1 + jj * hos_nx) * hos_nz +
                                           kk + 1] +
                                wx_hi * wy_hi * wz_lo *
                                    wdata
                                        .u[(ii + 1 + (jj + 1) * hos_nx) *
                                               hos_nz +
                                           kk] +
                                wx_hi * wy_hi * wz_hi *
                                    wdata
                                        .u[(ii + 1 + (jj + 1) * hos_nx) *
                                               hos_nz +
                                           kk + 1];
                            vel(i, j, k, 1) =
                                wx_lo * wy_lo * wz_lo *
                                    wdata.v[(ii + jj * hos_nx) * hos_nz + kk] +
                                wx_lo * wy_lo * wz_hi *
                                    wdata
                                        .v[(ii + jj * hos_nx) * hos_nz + kk +
                                           1] +
                                wx_lo * wy_hi * wz_lo *
                                    wdata
                                        .v[(ii + (jj + 1) * hos_nx) * hos_nz +
                                           kk] +
                                wx_lo * wy_hi * wz_hi *
                                    wdata
                                        .v[(ii + (jj + 1) * hos_nx) * hos_nz +
                                           kk + 1] +
                                wx_hi * wy_lo * wz_lo *
                                    wdata
                                        .v[(ii + 1 + jj * hos_nx) * hos_nz +
                                           kk] +
                                wx_hi * wy_lo * wz_hi *
                                    wdata
                                        .v[(ii + 1 + jj * hos_nx) * hos_nz +
                                           kk + 1] +
                                wx_hi * wy_hi * wz_lo *
                                    wdata
                                        .v[(ii + 1 + (jj + 1) * hos_nx) *
                                               hos_nz +
                                           kk] +
                                wx_hi * wy_hi * wz_hi *
                                    wdata
                                        .v[(ii + 1 + (jj + 1) * hos_nx) *
                                               hos_nz +
                                           kk + 1];
                            vel(i, j, k, 2) =
                                wx_lo * wy_lo * wz_lo *
                                    wdata.w[(ii + jj * hos_nx) * hos_nz + kk] +
                                wx_lo * wy_lo * wz_hi *
                                    wdata
                                        .w[(ii + jj * hos_nx) * hos_nz + kk +
                                           1] +
                                wx_lo * wy_hi * wz_lo *
                                    wdata
                                        .w[(ii + (jj + 1) * hos_nx) * hos_nz +
                                           kk] +
                                wx_lo * wy_hi * wz_hi *
                                    wdata
                                        .w[(ii + (jj + 1) * hos_nx) * hos_nz +
                                           kk + 1] +
                                wx_hi * wy_lo * wz_lo *
                                    wdata
                                        .w[(ii + 1 + jj * hos_nx) * hos_nz +
                                           kk] +
                                wx_hi * wy_lo * wz_hi *
                                    wdata
                                        .w[(ii + 1 + jj * hos_nx) * hos_nz +
                                           kk + 1] +
                                wx_hi * wy_hi * wz_lo *
                                    wdata
                                        .w[(ii + 1 + (jj + 1) * hos_nx) *
                                               hos_nz +
                                           kk] +
                                wx_hi * wy_hi * wz_hi *
                                    wdata
                                        .w[(ii + 1 + (jj + 1) * hos_nx) *
                                               hos_nz +
                                           kk + 1];
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

        return;

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

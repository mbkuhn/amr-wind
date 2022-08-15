#ifndef HOS_WAVES_OPS_H
#define HOS_WAVES_OPS_H

#include "amr-wind/ocean_waves/regular_waves/HOSWaves.H"
#include "amr-wind/ocean_waves/OceanWavesOps.H"
#include "amr-wind/ocean_waves/OceanWaves.H"
#include "amr-wind/ocean_waves/regular_waves/regular_waves_ops.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/physics/multiphase/MultiPhase.H"
#include "amr-wind/core/FieldRepo.H"

namespace amr_wind {
namespace ocean_waves {

namespace ops {

void ReadHOSFileLev(
    int lev,
    std::string HOS_prefix,
    int HOS_n,
    amrex::Real& HOS_t,
    amrex::Real& HOS_dt,
    RealList& eta,
    RealList& u,
    RealList& v,
    RealList& w,
    int& HOS_nx,
    amrex::Real& HOS_Lx,
    int& HOS_ny,
    amrex::Real& HOS_Ly,
    int& HOS_nz,
    amrex::Real& HOS_zmin,
    amrex::Real& HOS_zmax)
{
    // Convert file string to stream
    std::stringstream fname;
    fname << HOS_prefix << "_lev" << lev << "_" << HOS_n << ".txt";
    std::ifstream is(fname.str());
    // Read metadata from file
    std::string tmp;
    // Get initial time
    std::getline(is, tmp, '=');
    std::getline(is, tmp);
    HOS_t = std::stof(tmp);
    // Get dt
    std::getline(is, tmp, '=');
    std::getline(is, tmp);
    HOS_dt = std::stof(tmp);
    // Get nx, Lx
    std::getline(is, tmp, '=');
    std::getline(is, tmp, ',');
    HOS_nx = std::stoi(tmp);
    std::getline(is, tmp, '=');
    std::getline(is, tmp);
    HOS_Lx = std::stof(tmp);
    // Get ny, Ly
    std::getline(is, tmp, '=');
    std::getline(is, tmp, ',');
    HOS_ny = std::stoi(tmp);
    std::getline(is, tmp, '=');
    std::getline(is, tmp);
    HOS_Ly = std::stof(tmp);
    // Get nz, zmin, zmax
    std::getline(is, tmp, '=');
    std::getline(is, tmp, ',');
    HOS_nz = std::stoi(tmp);
    std::getline(is, tmp, '=');
    std::getline(is, tmp, ',');
    HOS_zmin = std::stof(tmp);
    std::getline(is, tmp, '=');
    std::getline(is, tmp);
    HOS_zmax = std::stof(tmp);

    // Modify Lx, Ly to account for node - cell difference
    /*HOS_Lx *= (((amrex::Real)HOS_nx) / (amrex::Real)(HOS_nx - std::pow(2,lev)));
    HOS_Ly *= (((amrex::Real)HOS_ny) / (amrex::Real)(HOS_ny - std::pow(2,lev)));*/

    // Allocate arrays for storage
    eta.resize(HOS_nx * HOS_ny);
    u.resize(HOS_nx * HOS_ny * HOS_nz);
    v.resize(HOS_nx * HOS_ny * HOS_nz);
    w.resize(HOS_nx * HOS_ny * HOS_nz);
    // Skip key
    std::getline(is, tmp);
    // Read interface heights and velocities
    for (int ilat = 0; ilat < HOS_nx * HOS_ny; ++ilat) {
        // Get eta for current point
        is >> eta[ilat];
        // Get u, v, w for full depth of 2D point
        for (int ivert = 0; ivert < HOS_nz; ++ivert) {
            is >> u[ilat * HOS_nz + ivert] >> v[ilat * HOS_nz + ivert] >>
                w[ilat * HOS_nz + ivert];
        }
    }
}

void ReadHOSFileLev(
    int lev,
    std::string HOS_prefix,
    int HOS_n,
    RealList& eta,
    RealList& u,
    RealList& v,
    RealList& w,
    int& HOS_nx,
    amrex::Real& HOS_Lx,
    int& HOS_ny,
    amrex::Real& HOS_Ly,
    int& HOS_nz,
    amrex::Real& HOS_zmin,
    amrex::Real& HOS_zmax)
{
    amrex::Real tmp1, tmp2;
    ReadHOSFileLev(
        lev, HOS_prefix, HOS_n, tmp1, tmp2, eta, u, v, w, HOS_nx, HOS_Lx,
        HOS_ny, HOS_Ly, HOS_nz, HOS_zmin, HOS_zmax);
}

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

        // Declare fields for HOS
        data.sim().repo().declare_field("hos_levelset", 1, 1, 1);
        data.sim().repo().declare_field("hos_velocity", AMREX_SPACEDIM, 1, 1);

    }
};

template <>
struct InitDataOp<HOSWaves>
{
    void operator()(HOSWaves::DataType& data)
    {
        auto& wdata = data.meta();

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
        // Set up variables that are re-written at each level
        RealList eta, u, v, w;
        int HOS_nx, HOS_ny, HOS_nz;
        amrex::Real HOS_Lx, HOS_Ly, HOS_zmin, HOS_zmax;
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& problo = geom[lev].ProbLoArray();
            const auto& probhi = geom[lev].ProbHiArray();
            const auto& dx = geom[lev].CellSizeArray();
            // Read HOS data at current level
            ReadHOSFileLev(
                lev, wdata.HOS_prefix, wdata.HOS_n, wdata.HOS_t, wdata.HOS_dt,
                eta, u, v, w, HOS_nx, HOS_Lx, HOS_ny, HOS_Ly, HOS_nz, HOS_zmin,
                HOS_zmax);

            // Check if current dimensions are compatible
            if (problo[0] < -1e-6 || probhi[0] > HOS_Lx * (1.0 + 1e-6) ||
                problo[1] < -1e-6 || probhi[1] > HOS_Ly * (1.0 + 1e-6)) {
                amrex::Abort(
                    "HOS OceanWaves: Lateral dimensions incompatible, level " +
                    std::to_string(lev));
            }
            for (amrex::MFIter mfi(m_ow_levelset(lev)); mfi.isValid(); ++mfi) {
                auto phi = m_ow_levelset(lev).array(mfi);
                auto vel = velocity(lev).array(mfi);
                auto ow_vel = m_ow_velocity(lev).array(mfi);
                auto volfrac = vof(lev).array(mfi);
                auto rho = density(lev).array(mfi);
                const amrex::Real zsl = wdata.zsl;
                const amrex::Real HOS_dx = HOS_Lx / HOS_nx;
                const amrex::Real HOS_dy = HOS_Ly / HOS_ny;
                const amrex::Real HOS_dz = (HOS_zmax - HOS_zmin) / HOS_nz;
                const auto& gbx = mfi.growntilebox();
                const auto& vbx = mfi.validbox();
                amrex::ParallelFor(
                    gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                        amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                        amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                        // Adjust x,y for periodicity:
                        if (x > probhi[0]) x -= (probhi[0] - problo[0]);
                        if (y > probhi[1]) y -= (probhi[1] - problo[1]);
                        if (x < problo[0]) x += (probhi[0] - problo[0]);
                        if (y < problo[1]) y += (probhi[1] - problo[1]);
                        // Determine lateral indices for interpolation
                        // It's best to have the grids line up directly
                        int ii = 0;
                        amrex::Real HOS_x = HOS_dx * (ii + 0.5);
                        while (HOS_x < x) {
                            ++ii;
                            HOS_x = HOS_dx * (ii + 0.5);
                        }
                        int jj = 0;
                        amrex::Real HOS_y = HOS_dy * (jj + 0.5);
                        while (HOS_y < y) {
                            ++jj;
                            HOS_y = HOS_dy * (jj + 0.5);
                        }

                        if (z > HOS_zmax || z < HOS_zmin) {
                            phi(i, j, k) =
                                eta[ii + jj * HOS_nx] + zsl - z;
                            // Assume velocities are zero outside range
                            vel(i, j, k, 0) = 0;
                            vel(i, j, k, 1) = 0;
                            vel(i, j, k, 2) = 0;
                        } else {
                            // Determine vertical indices
                            int kk = 0;
                            amrex::Real HOS_z =
                                HOS_zmin + HOS_dz * (kk + 0.5);
                            while (HOS_z < z) {
                                ++kk;
                                HOS_z = HOS_zmin + HOS_dz * (kk + 0.5);
                            }
                            // Shift indices left
                            ii = ii - 1;
                            jj = jj - 1;
                            kk = kk - 1;
                            // Modify indices to avoid going out of bounds
                            ii = amrex::max(0,amrex::min(HOS_nx-1,ii));
                            jj = amrex::max(0,amrex::min(HOS_ny-1,jj));
                            kk = amrex::max(0,amrex::min(HOS_nz-1,kk));
                            // Reset locations
                            HOS_x = HOS_dx * (ii + 0.5);
                            HOS_y = HOS_dy * (jj + 0.5);
                            HOS_z = HOS_zmin + HOS_dz * (kk + 0.5);
                            // Interpolation factors
                            amrex::Real wx_hi = (x - HOS_x) / HOS_dx;
                            amrex::Real wy_hi = (y - HOS_y) / HOS_dy;
                            amrex::Real wz_hi = (z - HOS_z) / HOS_dz;
                            amrex::Real wx_lo = 1.0 - wx_hi;
                            amrex::Real wy_lo = 1.0 - wy_hi;
                            amrex::Real wz_lo = 1.0 - wz_hi;

                            // Get data
                            const amrex::Real eta_00 =
                                eta[ii + jj * HOS_nx];
                            const amrex::Real eta_01 =
                                eta[ii + (jj + 1) * HOS_nx];
                            const amrex::Real eta_10 =
                                eta[ii + 1 + jj * HOS_nx];
                            const amrex::Real eta_11 =
                                eta[ii + 1 + (jj + 1) * HOS_nx];
                            const amrex::Real eta_ = wx_lo * wy_lo * eta_00 +
                                                     wx_lo * wy_hi * eta_01 +
                                                     wx_hi * wy_lo * eta_10 +
                                                     wx_hi * wy_hi * eta_11;

                            phi(i, j, k) = eta_ + zsl - z;

                            // Set up indices for clarity
                            const int i000 = (ii + jj * HOS_nx) * HOS_nz + kk;
                            const int i001 =
                                (ii + jj * HOS_nx) * HOS_nz + kk + 1;
                            const int i010 =
                                (ii + (jj + 1) * HOS_nx) * HOS_nz + kk;
                            const int i100 =
                                (ii + 1 + jj * HOS_nx) * HOS_nz + kk;
                            const int i011 =
                                (ii + (jj + 1) * HOS_nx) * HOS_nz + kk + 1;
                            const int i101 =
                                (ii + 1 + jj * HOS_nx) * HOS_nz + kk + 1;
                            const int i110 =
                                (ii + 1 + (jj + 1) * HOS_nx) * HOS_nz + kk;
                            const int i111 =
                                (ii + 1 + (jj + 1) * HOS_nx) * HOS_nz + kk + 1;
                            vel(i, j, k, 0) =
                                wx_lo * wy_lo * wz_lo * u[i000] +
                                wx_lo * wy_lo * wz_hi * u[i001] +
                                wx_lo * wy_hi * wz_lo * u[i010] +
                                wx_lo * wy_hi * wz_hi * u[i011] +
                                wx_hi * wy_lo * wz_lo * u[i100] +
                                wx_hi * wy_lo * wz_hi * u[i101] +
                                wx_hi * wy_hi * wz_lo * u[i110] +
                                wx_hi * wy_hi * wz_hi * u[i111];
                            vel(i, j, k, 1) =
                                wx_lo * wy_lo * wz_lo * v[i000] +
                                wx_lo * wy_lo * wz_hi * v[i001] +
                                wx_lo * wy_hi * wz_lo * v[i010] +
                                wx_lo * wy_hi * wz_hi * v[i011] +
                                wx_hi * wy_lo * wz_lo * v[i100] +
                                wx_hi * wy_lo * wz_hi * v[i101] +
                                wx_hi * wy_hi * wz_lo * v[i110] +
                                wx_hi * wy_hi * wz_hi * v[i111];
                            vel(i, j, k, 2) =
                                wx_lo * wy_lo * wz_lo * w[i000] +
                                wx_lo * wy_lo * wz_hi * w[i001] +
                                wx_lo * wy_hi * wz_lo * w[i010] +
                                wx_lo * wy_hi * wz_hi * w[i011] +
                                wx_hi * wy_lo * wz_lo * w[i100] +
                                wx_hi * wy_lo * wz_hi * w[i101] +
                                wx_hi * wy_hi * wz_lo * w[i110] +
                                wx_hi * wy_hi * wz_hi * w[i111];

                            // Set velocity to zero if no liquid present
                            if (phi(i, j, k) + 0.5 * dx[2] < 0.0) {
                                vel(i, j, k, 0) = 0.0;
                                vel(i, j, k, 1) = 0.0;
                                vel(i, j, k, 2) = 0.0;
                            }
                        }
                        // ow velocity should initially be in sync with velocity
                        ow_vel(i, j, k, 0) = vel(i, j, k, 0);
                        ow_vel(i, j, k, 1) = vel(i, j, k, 1);
                        ow_vel(i, j, k, 2) = vel(i, j, k, 2);
                    });

                const amrex::Real eps = 2. * std::cbrt(dx[0] * dx[1] * dx[2]);
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        multiphase::levelset_to_vof(i, j, k, eps, phi, volfrac);
                        rho(i, j, k) = rho1 * volfrac(i, j, k) +
                                       rho2 * (1.0 - volfrac(i, j, k));
                    });
            }
        }
        // Fill boundaries with vof, density
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
        bool read_flag = false;
        if (time > wdata.HOS_t) {
            // Increment time step
            ++wdata.HOS_n;
            // Increment time
            wdata.HOS_t += wdata.HOS_dt;
            // Flag to store new data
            read_flag = true;
        }

        // Update ow fields every time
        auto& m_ow_levelset = sim.repo().get_field("ow_levelset");
        auto& m_ow_velocity = sim.repo().get_field("ow_velocity");
        // Update HOS fields when necessary
        auto& hos_levelset = sim.repo().get_field("hos_levelset");
        auto& hos_velocity = sim.repo().get_field("hos_velocity");

        auto nlevels = sim.repo().num_active_levels();
        auto geom = sim.mesh().Geom();
        // Set up variables that are re-written at each level
        RealList eta, u, v, w;
        int HOS_nx, HOS_ny, HOS_nz;
        amrex::Real HOS_Lx, HOS_Ly, HOS_zmin, HOS_zmax;
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& problo = geom[lev].ProbLoArray();
            const auto& dx = geom[lev].CellSizeArray();
            // Read HOS data at current level
            if (read_flag) {
                ReadHOSFileLev(
                    lev, wdata.HOS_prefix, wdata.HOS_n, eta, u, v, w, HOS_nx,
                    HOS_Lx, HOS_ny, HOS_Ly, HOS_nz, HOS_zmin, HOS_zmax);
            }
            for (amrex::MFIter mfi(m_ow_levelset(lev)); mfi.isValid(); ++mfi) {
                auto phi = m_ow_levelset(lev).array(mfi);
                auto vel = m_ow_velocity(lev).array(mfi);
                auto HOS_phi = hos_levelset(lev).array(mfi);
                auto HOS_vel = hos_velocity(lev).array(mfi);

                const amrex::Real zsl = wdata.zsl;
                const amrex::Real HOS_dx = HOS_Lx / HOS_nx;
                const amrex::Real HOS_dy = HOS_Ly / HOS_ny;
                const amrex::Real HOS_dz = (HOS_zmax - HOS_zmin) / HOS_nz;
                const amrex::Real HOS_t = wdata.HOS_t;
                const amrex::Real t_last = wdata.t_last;
                const auto& vbx = mfi.validbox();
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        // Update fields if data has been just read
                        if (read_flag) {
                            const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                            const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                            const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                            // Determine lateral indices for interpolation
                            // It's best to have the grids line up directly
                            int ii = 0;
                            amrex::Real HOS_x = HOS_dx * (ii + 0.5);
                            while (HOS_x < x) {
                                ++ii;
                                HOS_x = HOS_dx * (ii + 0.5);
                            }
                            int jj = 0;
                            amrex::Real HOS_y = HOS_dy * (jj + 0.5);
                            while (HOS_y < y) {
                                ++jj;
                                HOS_y = HOS_dy * (jj + 0.5);
                            }

                            if (z > HOS_zmax || z < HOS_zmin) {
                                HOS_phi(i, j, k) =
                                    eta[ii + jj * HOS_nx] + zsl - z;
                                // Assume velocities are zero outside range
                                HOS_vel(i, j, k, 0) = 0;
                                HOS_vel(i, j, k, 1) = 0;
                                HOS_vel(i, j, k, 2) = 0;
                            } else {
                                // Determine vertical indices
                                int kk = 0;
                                amrex::Real HOS_z =
                                    HOS_zmin + HOS_dz * (kk + 0.5);
                                while (HOS_z < z) {
                                    ++kk;
                                    HOS_z = HOS_zmin + HOS_dz * (kk + 0.5);
                                }
                                // Shift indices left
                                ii = ii - 1;
                                jj = jj - 1;
                                kk = kk - 1;
                                // Modify indices to avoid going out of bounds
                                ii = amrex::max(0, amrex::min(HOS_nx - 1, ii));
                                jj = amrex::max(0, amrex::min(HOS_ny - 1, jj));
                                kk = amrex::max(0, amrex::min(HOS_nz - 1, kk));
                                // Reset locations
                                HOS_x = HOS_dx * (ii + 0.5);
                                HOS_y = HOS_dy * (jj + 0.5);
                                HOS_z = HOS_zmin + HOS_dz * (kk + 0.5);
                                // Interpolation factors
                                amrex::Real wx_hi = (x - HOS_x) / HOS_dx;
                                amrex::Real wy_hi = (y - HOS_y) / HOS_dy;
                                amrex::Real wz_hi = (z - HOS_z) / HOS_dz;
                                amrex::Real wx_lo = 1.0 - wx_hi;
                                amrex::Real wy_lo = 1.0 - wy_hi;
                                amrex::Real wz_lo = 1.0 - wz_hi;

                                // Get data
                                const amrex::Real eta_00 =
                                    eta[ii + jj * HOS_nx];
                                const amrex::Real eta_01 =
                                    eta[ii + (jj + 1) * HOS_nx];
                                const amrex::Real eta_10 =
                                    eta[ii + 1 + jj * HOS_nx];
                                const amrex::Real eta_11 =
                                    eta[ii + 1 + (jj + 1) * HOS_nx];
                                const amrex::Real eta_ =
                                    wx_lo * wy_lo * eta_00 +
                                    wx_lo * wy_hi * eta_01 +
                                    wx_hi * wy_lo * eta_10 +
                                    wx_hi * wy_hi * eta_11;

                                HOS_phi(i, j, k) = eta_ + zsl - z;

                                // Set up indices for clarity
                                const int i000 =
                                    (ii + jj * HOS_nx) * HOS_nz + kk;
                                const int i001 =
                                    (ii + jj * HOS_nx) * HOS_nz + kk + 1;
                                const int i010 =
                                    (ii + (jj + 1) * HOS_nx) * HOS_nz + kk;
                                const int i100 =
                                    (ii + 1 + jj * HOS_nx) * HOS_nz + kk;
                                const int i011 =
                                    (ii + (jj + 1) * HOS_nx) * HOS_nz + kk + 1;
                                const int i101 =
                                    (ii + 1 + jj * HOS_nx) * HOS_nz + kk + 1;
                                const int i110 =
                                    (ii + 1 + (jj + 1) * HOS_nx) * HOS_nz + kk;
                                const int i111 =
                                    (ii + 1 + (jj + 1) * HOS_nx) * HOS_nz + kk +
                                    1;
                                HOS_vel(i,j,k,0) = wx_lo * wy_lo * wz_lo * u[i000] +
                                        wx_lo * wy_lo * wz_hi * u[i001] +
                                        wx_lo * wy_hi * wz_lo * u[i010] +
                                        wx_lo * wy_hi * wz_hi * u[i011] +
                                        wx_hi * wy_lo * wz_lo * u[i100] +
                                        wx_hi * wy_lo * wz_hi * u[i101] +
                                        wx_hi * wy_hi * wz_lo * u[i110] +
                                        wx_hi * wy_hi * wz_hi * u[i111];
                                HOS_vel(i,j,k,1) = wx_lo * wy_lo * wz_lo * v[i000] +
                                        wx_lo * wy_lo * wz_hi * v[i001] +
                                        wx_lo * wy_hi * wz_lo * v[i010] +
                                        wx_lo * wy_hi * wz_hi * v[i011] +
                                        wx_hi * wy_lo * wz_lo * v[i100] +
                                        wx_hi * wy_lo * wz_hi * v[i101] +
                                        wx_hi * wy_hi * wz_lo * v[i110] +
                                        wx_hi * wy_hi * wz_hi * v[i111];
                                HOS_vel(i,j,k,2) = wx_lo * wy_lo * wz_lo * w[i000] +
                                        wx_lo * wy_lo * wz_hi * w[i001] +
                                        wx_lo * wy_hi * wz_lo * w[i010] +
                                        wx_lo * wy_hi * wz_hi * w[i011] +
                                        wx_hi * wy_lo * wz_lo * w[i100] +
                                        wx_hi * wy_lo * wz_hi * w[i101] +
                                        wx_hi * wy_hi * wz_lo * w[i110] +
                                        wx_hi * wy_hi * wz_hi * w[i111];
                            }
                        }
                        // Interpolate temporally every time
                        phi(i, j, k) += (HOS_phi(i, j, k) - phi(i, j, k)) /
                                        (HOS_t - t_last) * (time - t_last);
                        vel(i, j, k, 0) +=
                            (HOS_vel(i, j, k, 0) - vel(i, j, k, 0)) /
                            (HOS_t - t_last) * (time - t_last);
                        vel(i, j, k, 1) +=
                            (HOS_vel(i, j, k, 1) - vel(i, j, k, 1)) /
                            (HOS_t - t_last) * (time - t_last);
                        vel(i, j, k, 2) +=
                            (HOS_vel(i, j, k, 2) - vel(i, j, k, 2)) /
                            (HOS_t - t_last) * (time - t_last);
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

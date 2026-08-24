#include "src/CFDSim.H"
#include "src/boundary_conditions/field_boundary_fill/Flather.H"
#include "src/boundary_conditions/field_boundary_fill/FillFlather.H"
#include "src/utilities/index_operations.H"
#include "src/utilities/constants.H"
#include "AMReX_GpuAtomic.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include <algorithm>
#include <cmath>

using namespace amrex::literals;

namespace kynema_sgf {

Flather::Flather(CFDSim& sim)
    : m_sim(sim)
    , m_time(m_sim.time())
    , m_repo(m_sim.repo())
    , m_mesh(m_sim.mesh())
    , m_velocity(m_sim.repo().get_field("velocity"))
    , m_vof(m_sim.repo().get_field("vof"))
{
    // amrex::ParmParse pp(identifier());

    if (!m_repo.field_exists("vof")) {
        amrex::Abort("Flather BC requires the vof field");
    }
    if (m_repo.int_field_exists("terrain_blank")) {
        m_terrain_blank = &m_repo.get_int_field("terrain_blank");
    }

    amrex::ParmParse pp("incflo");
    pp.queryarr("gravity", m_gravity);

    m_xlo_uvof_avg.resize(0, m_mesh.Geom());
    m_xhi_uvof_avg.resize(0, m_mesh.Geom());
    m_ylo_uvof_avg.resize(1, m_mesh.Geom());
    m_yhi_uvof_avg.resize(1, m_mesh.Geom());
    m_xlo_bnd_uvof_avg.resize(0, m_mesh.Geom());
    m_xhi_bnd_uvof_avg.resize(0, m_mesh.Geom());
    m_ylo_bnd_uvof_avg.resize(1, m_mesh.Geom());
    m_yhi_bnd_uvof_avg.resize(1, m_mesh.Geom());
}

void Flather::post_init_actions()
{
    m_velocity.register_fill_patch_op<FillFlather>(m_mesh, m_time, *this);
    compute_boundary_z_averages();
}

void Flather::pre_advance_work() { compute_boundary_z_averages(); }

void Flather::accumulate_boundary(
    const int lev,
    const int idir,
    const bool is_low,
    MultiLevelVector& out_uvec,
    MultiLevelVector& out_hvec,
    const bool sample_boundary) const
{
    AMREX_ALWAYS_ASSERT(idir == 0 || idir == 1);

    auto& avg_h = out_uvec.host_data(lev);
    auto& dist_h = out_hvec.host_data(lev);
    const int nline = static_cast<int>(avg_h.size());

    amrex::Vector<amrex::Real> uvof_sum_h(nline, 0.0_rt);
    amrex::Vector<amrex::Real> vof_sum_h(nline, 0.0_rt);
    amrex::Gpu::DeviceVector<amrex::Real> uvof_sum_d(nline, 0.0_rt);
    amrex::Gpu::DeviceVector<amrex::Real> vof_sum_d(nline, 0.0_rt);

    const auto& geom = m_mesh.Geom(lev);
    const auto& dz = geom.CellSizeArray()[2];
    const auto& dom = geom.Domain();
    const int bidx = is_low ? dom.smallEnd(idir) : dom.bigEnd(idir);
    const int off = dom.smallEnd(idir);
    const int shift_to_boundary = sample_boundary ? (is_low ? -1 : 1) : 0;
    if (shift_to_boundary != 0) {
        AMREX_ALWAYS_ASSERT(m_velocity.num_grow()[idir] > 0);
        AMREX_ALWAYS_ASSERT(m_vof.num_grow()[idir] > 0);
    }
    const amrex::Real tiny = constants::TIGHT_TOL;

    const auto& vel_mf = m_velocity(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(vel_mf, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        amrex::Box bx = mfi.tilebox() & dom;
        if (!bx.ok()) {
            continue;
        }
        if (bidx < bx.smallEnd(idir) || bidx > bx.bigEnd(idir)) {
            continue;
        }

        bx.setSmall(idir, bidx);
        bx.setBig(idir, bidx);

        const auto vel_arr = vel_mf.const_array(mfi);
        const auto vof_arr = m_vof(lev).const_array(mfi);
        const bool use_terrain = (m_terrain_blank != nullptr);
        const auto terrain_blank_arr =
            use_terrain ? (*m_terrain_blank)(lev).const_array(mfi)
                        : amrex::Array4<int const>();

        amrex::Real* uvof_sum = uvof_sum_d.data();
        amrex::Real* vof_sum = vof_sum_d.data();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const int ii = (idir == 0) ? (i + shift_to_boundary) : i;
            const int jj = (idir == 1) ? (j + shift_to_boundary) : j;

            if (use_terrain && terrain_blank_arr(i, j, k) != 0) {
                return;
            }
            const int idx = (idir == 0) ? (i - off) : (j - off);
            const amrex::Real vf =
                amrex::max(0.0_rt, amrex::min(1.0_rt, vof_arr(ii, jj, k)));
            amrex::Gpu::Atomic::Add(
                &uvof_sum[idx], vel_arr(ii, jj, k, idir) * vf * dz);
            amrex::Gpu::Atomic::Add(&vof_sum[idx], vf * dz);
        });
    }

    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, uvof_sum_d.begin(), uvof_sum_d.end(),
        uvof_sum_h.begin());
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, vof_sum_d.begin(), vof_sum_d.end(),
        vof_sum_h.begin());

    amrex::ParallelDescriptor::ReduceRealSum(uvof_sum_h.data(), nline);
    amrex::ParallelDescriptor::ReduceRealSum(vof_sum_h.data(), nline);

    for (int n = 0; n < nline; ++n) {
        avg_h[n] = (vof_sum_h[n] > tiny) ? (uvof_sum_h[n] / vof_sum_h[n])
                                         : 0.0_rt;
        dist_h[n] = vof_sum_h[n];
    }
}

void Flather::compute_boundary_z_averages()
{
    BL_PROFILE("kynema-sgf::Flather::compute_boundary_z_averages");

    const int nlevels = m_repo.num_active_levels();
    const int nlevels_geom = static_cast<int>(m_mesh.Geom().size());
    if (m_xlo_uvof_avg.size() != nlevels_geom) {
        m_xlo_uvof_avg.resize(0, m_mesh.Geom());
        m_xhi_uvof_avg.resize(0, m_mesh.Geom());
        m_ylo_uvof_avg.resize(1, m_mesh.Geom());
        m_yhi_uvof_avg.resize(1, m_mesh.Geom());
        m_xlo_bnd_uvof_avg.resize(0, m_mesh.Geom());
        m_xhi_bnd_uvof_avg.resize(0, m_mesh.Geom());
        m_ylo_bnd_uvof_avg.resize(1, m_mesh.Geom());
        m_yhi_bnd_uvof_avg.resize(1, m_mesh.Geom());
    }

    for (int lev = 0; lev < nlevels; ++lev) {
        this->accumulate_boundary(
            lev, 0, true, m_xlo_uvof_avg, m_xlo_h_avg, false);
        this->accumulate_boundary(
            lev, 0, false, m_xhi_uvof_avg, m_xhi_h_avg, false);
        this->accumulate_boundary(
            lev, 1, true, m_ylo_uvof_avg, m_ylo_h_avg, false);
        this->accumulate_boundary(
            lev, 1, false, m_yhi_uvof_avg, m_yhi_h_avg, false);
        this->accumulate_boundary(
            lev, 0, true, m_xlo_bnd_uvof_avg, m_xlo_bnd_h_avg, true);
        this->accumulate_boundary(
            lev, 0, false, m_xhi_bnd_uvof_avg, m_xhi_bnd_h_avg, true);
        this->accumulate_boundary(
            lev, 1, true, m_ylo_bnd_uvof_avg, m_ylo_bnd_h_avg, true);
        this->accumulate_boundary(
            lev, 1, false, m_yhi_bnd_uvof_avg, m_yhi_bnd_h_avg, true);
    }

    m_xlo_uvof_avg.copy_host_to_device();
    m_xhi_uvof_avg.copy_host_to_device();
    m_ylo_uvof_avg.copy_host_to_device();
    m_yhi_uvof_avg.copy_host_to_device();
    m_xlo_h_avg.copy_host_to_device();
    m_xhi_h_avg.copy_host_to_device();
    m_ylo_h_avg.copy_host_to_device();
    m_yhi_h_avg.copy_host_to_device();

    m_xlo_bnd_uvof_avg.copy_host_to_device();
    m_xhi_bnd_uvof_avg.copy_host_to_device();
    m_ylo_bnd_uvof_avg.copy_host_to_device();
    m_yhi_bnd_uvof_avg.copy_host_to_device();
    m_xlo_bnd_h_avg.copy_host_to_device();
    m_xhi_bnd_h_avg.copy_host_to_device();
    m_ylo_bnd_h_avg.copy_host_to_device();
    m_yhi_bnd_h_avg.copy_host_to_device();
}

void Flather::set_velocity(
    const int lev,
    const amrex::Real /*time*/,
    const Field& fld,
    amrex::MultiFab& mfab,
    const int dcomp,
    const int orig_comp) const
{
    BL_PROFILE("kynema-sgf::Flather::set_velocity");

    const auto& geom = m_mesh.Geom(lev);
    const auto& bctype = fld.bc_type();
    const int nghost = 1;
    const auto& domain = geom.growPeriodicDomain(nghost);

    const amrex::Real tiny = constants::TIGHT_TOL;
    const auto grav_z = -m_gravity[2];

    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if ((bctype[ori] != BC::mass_inflow) &&
            (bctype[ori] != BC::mass_inflow_outflow)) {
            continue;
        }

        const int idir = ori.coordDir();
        const auto& dbx = ori.isLow() ? amrex::adjCellLo(domain, idir, nghost)
                                      : amrex::adjCellHi(domain, idir, nghost);
        const auto shift_to_interior =
            amrex::IntVect::TheDimensionVector(idir) * (ori.isLow() ? 1 : -1);
        const bool use_x = (idir == 0);
        const bool use_y = (idir == 1);
        const int xoff = geom.Domain().smallEnd(0);
        const int yoff = geom.Domain().smallEnd(1);
        const amrex::Real* xlo_uavg = m_xlo_uvof_avg.device_data(lev).data();
        const amrex::Real* xhi_uavg = m_xhi_uvof_avg.device_data(lev).data();
        const amrex::Real* ylo_uavg = m_ylo_uvof_avg.device_data(lev).data();
        const amrex::Real* yhi_uavg = m_yhi_uvof_avg.device_data(lev).data();
        const amrex::Real* xlo_havg = m_xlo_h_avg.device_data(lev).data();
        const amrex::Real* xhi_havg = m_xhi_h_avg.device_data(lev).data();
        const amrex::Real* ylo_havg = m_ylo_h_avg.device_data(lev).data();
        const amrex::Real* yhi_havg = m_yhi_h_avg.device_data(lev).data();
        const amrex::Real* xlo_bnd_uavg =
            m_xlo_bnd_uvof_avg.device_data(lev).data();
        const amrex::Real* xhi_bnd_uavg =
            m_xhi_bnd_uvof_avg.device_data(lev).data();
        const amrex::Real* ylo_bnd_uavg =
            m_ylo_bnd_uvof_avg.device_data(lev).data();
        const amrex::Real* yhi_bnd_uavg =
            m_yhi_bnd_uvof_avg.device_data(lev).data();
        const amrex::Real* xlo_bnd_havg =
            m_xlo_bnd_h_avg.device_data(lev).data();
        const amrex::Real* xhi_bnd_havg =
            m_xhi_bnd_h_avg.device_data(lev).data();
        const amrex::Real* ylo_bnd_havg =
            m_ylo_bnd_h_avg.device_data(lev).data();
        const amrex::Real* yhi_bnd_havg =
            m_yhi_bnd_h_avg.device_data(lev).data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (false)
#endif
        for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
            auto gbx = amrex::grow(mfi.validbox(), nghost);
            const auto& bx =
                utils::face_aware_boundary_box_intersection(gbx, dbx, ori);
            if (!bx.ok()) {
                continue;
            }

            const auto& arr = mfab[mfi].array();
            const int numcomp = mfab.nComp();

            const auto vof_arr = m_vof(lev).const_array(mfi);
            const bool use_terrain = (m_terrain_blank != nullptr);
            const auto terrain_blank_arr =
                use_terrain ? (*m_terrain_blank)(lev).const_array(mfi)
                            : amrex::Array4<int const>();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const amrex::IntVect iv{i, j, k};
                const amrex::IntVect iv_adj = iv + shift_to_interior;

                for (int n = 0; n < numcomp; ++n) {
                    amrex::Real boundary_val = arr(iv, dcomp + n);
                    amrex::Real interior_val = arr(iv_adj, dcomp + n);
                    amrex::Real boundary_h = 0.0_rt;
                    amrex::Real interior_h = 0.0_rt;
                    if (use_x && (orig_comp + n == 0)) {
                        interior_val = ori.isLow() ? xlo_uavg[iv_adj[0] - xoff]
                                                   : xhi_uavg[iv_adj[0] - xoff];
                        interior_h = ori.isLow() ? xlo_havg[iv_adj[0] - xoff]
                                                 : xhi_havg[iv_adj[0] - xoff];
                        boundary_val = ori.isLow()
                                           ? xlo_bnd_uavg[iv_adj[0] - xoff]
                                           : xhi_bnd_uavg[iv_adj[0] - xoff];
                        boundary_h = ori.isLow()
                                         ? xlo_bnd_havg[iv_adj[0] - xoff]
                                         : xhi_bnd_havg[iv_adj[0] - xoff];
                    } else if (use_y && (orig_comp + n == 1)) {
                        interior_val = ori.isLow() ? ylo_uavg[iv_adj[1] - yoff]
                                                   : yhi_uavg[iv_adj[1] - yoff];
                        interior_h = ori.isLow() ? ylo_havg[iv_adj[1] - yoff]
                                                 : yhi_havg[iv_adj[1] - yoff];
                        boundary_val = ori.isLow()
                                           ? ylo_bnd_uavg[iv_adj[1] - yoff]
                                           : yhi_bnd_uavg[iv_adj[1] - yoff];
                        boundary_h = ori.isLow()
                                         ? ylo_bnd_havg[iv_adj[1] - yoff]
                                         : yhi_bnd_havg[iv_adj[1] - yoff];
                    }

                    // Set velocity to zero and skip for terrain
                    if (use_terrain && terrain_blank_arr(iv) != 0) {
                        arr(iv_adj, dcomp + n) = 0.0_rt;
                        continue;
                    }
                    // Check interior or boundary vof for liquid
                    const amrex::Real interior_vof = vof_arr(iv_adj);
                    const amrex::Real boundary_vof = vof_arr(iv);
                    // Do nothing here if not liquid
                    if ((interior_vof < tiny) && (boundary_vof < tiny)) {
                        continue;
                    }

                    // Wave speed
                    const amrex::Real c = std::sqrt(grav_z * interior_h);

                    const auto Flather_val =
                        boundary_val + (ori.isLow() ? -1.0_rt : 1.0_rt) * c /
                                           boundary_h *
                                           (interior_h - boundary_h);

                    arr(iv_adj, dcomp + n) =
                        arr(iv, dcomp + n) * (Flather_val / interior_val);
                }
            });
        }
    }
}

} // namespace kynema_sgf

#include "src/CFDSim.H"
#include "src/boundary_conditions/field_boundary_fill/Flather.H"
#include "src/boundary_conditions/field_boundary_fill/FillFlather.H"
#include "src/utilities/index_operations.H"
#include "src/utilities/constants.H"
#include "AMReX_MultiFabUtil.H"
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
    , m_u_mac(m_sim.repo().get_field("u_mac"))
    , m_v_mac(m_sim.repo().get_field("v_mac"))
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

    // Vector at the x boundary extends in y direction
    // Vector at the y boundary extends in x direction
    m_xlo_uvof_avg.resize(1, m_mesh.Geom());
    m_xhi_uvof_avg.resize(1, m_mesh.Geom());
    m_ylo_uvof_avg.resize(0, m_mesh.Geom());
    m_yhi_uvof_avg.resize(0, m_mesh.Geom());
    m_xlo_bnd_uvof_avg.resize(1, m_mesh.Geom());
    m_xhi_bnd_uvof_avg.resize(1, m_mesh.Geom());
    m_ylo_bnd_uvof_avg.resize(0, m_mesh.Geom());
    m_yhi_bnd_uvof_avg.resize(0, m_mesh.Geom());

    m_xlo_h_avg.resize(1, m_mesh.Geom());
    m_xhi_h_avg.resize(1, m_mesh.Geom());
    m_ylo_h_avg.resize(0, m_mesh.Geom());
    m_yhi_h_avg.resize(0, m_mesh.Geom());
    m_xlo_bnd_h_avg.resize(1, m_mesh.Geom());
    m_xhi_bnd_h_avg.resize(1, m_mesh.Geom());
    m_ylo_bnd_h_avg.resize(0, m_mesh.Geom());
    m_yhi_bnd_h_avg.resize(0, m_mesh.Geom());
}

void Flather::post_init_actions()
{
    m_velocity.add_fill_patch_op<FillFlather>(m_mesh, m_time, *this);
    compute_internal_z_averages();
}

void Flather::pre_advance_work() { compute_internal_z_averages(); }

void Flather::accumulate_boundary(
    int current_level,
    int idir,
    bool is_low,
    MultiLevelVector& out_uvec,
    MultiLevelVector& out_hvec,
    bool sample_boundary,
    FieldState fstate,
    bool use_mac_fields) const
{
    AMREX_ALWAYS_ASSERT(idir == 0 || idir == 1);

    auto& avg_h = out_uvec.host_data(current_level);
    auto& dist_h = out_hvec.host_data(current_level);
    const int nline = static_cast<int>(avg_h.size());

    amrex::Vector<amrex::Real> uvof_sum_h(nline, 0.0_rt);
    amrex::Gpu::DeviceVector<amrex::Real> uvof_sum_d(nline, 0.0_rt);
    amrex::Gpu::DeviceVector<amrex::Real> vof_sum_d(nline, 0.0_rt);

    const amrex::Real tiny = constants::TIGHT_TOL;

    // Loop through current level and all below
    for (int lev = current_level; lev >= 0; --lev) {

        const auto rr = m_mesh.refRatio(lev)[1 - idir];

        amrex::iMultiFab level_mask;
        if (lev < current_level) {
            level_mask = makeFineMask(
                m_mesh.boxArray(lev), m_mesh.DistributionMap(lev),
                m_mesh.boxArray(lev + 1), m_mesh.refRatio(lev), 1, 0);
        } else {
            level_mask.define(
                m_mesh.boxArray(lev), m_mesh.DistributionMap(lev), 1, 0,
                amrex::MFInfo());
            level_mask.setVal(1);
        }

        const auto& geom = m_mesh.Geom(lev);
        const auto& dz = geom.CellSizeArray()[2];
        const auto& dom = geom.Domain();
        const int bidx = is_low ? dom.smallEnd(idir) : dom.bigEnd(idir);
        const int shift_to_boundary = sample_boundary ? (is_low ? -1 : 1) : 0;
        const auto& src_vel =
            use_mac_fields ? ((idir == 0) ? m_u_mac : m_v_mac) : m_velocity;
        if (shift_to_boundary != 0) {
            AMREX_ALWAYS_ASSERT(src_vel.num_grow()[idir] > 0);
            AMREX_ALWAYS_ASSERT(m_vof.num_grow()[idir] > 0);
        }

        const auto& vel_mf =
            src_vel.state(use_mac_fields ? FieldState::New : fstate)(lev);

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

            // Limit box to along boundary
            bx.setSmall(idir, bidx);
            bx.setBig(idir, bidx);

            const auto vel_arr = vel_mf.const_array(mfi);
            // Due to the order that fillphysbc is called in prepare_boundaries
            // (velocity, then scalars), the vof data is not available at
            // alternate (nph) states when this is called.
            const auto vof_arr = m_vof(lev).const_array(mfi);
            const auto mask_arr = level_mask.const_array(mfi);
            const bool use_terrain = (m_terrain_blank != nullptr);
            const auto terrain_blank_arr =
                use_terrain ? (*m_terrain_blank)(lev).const_array(mfi)
                            : amrex::Array4<int const>();

            amrex::Real* uvof_sum = uvof_sum_d.data();
            amrex::Real* vof_sum = vof_sum_d.data();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const int ii = (idir == 0) ? (i + shift_to_boundary) : i;
                const int jj = (idir == 1) ? (j + shift_to_boundary) : j;
                const int ii_v = ii + static_cast<int>(idir == 0 && !is_low);
                const int jj_v = jj + static_cast<int>(idir == 1 && !is_low);

                if (use_terrain && terrain_blank_arr(i, j, k) != 0) {
                    return;
                }
                // This index is tangent to the boundary
                const int idx_lev = (idir == 0) ? j : i;
                // Convert to current level indices
                const int idx_min =
                    idx_lev + idx_lev * (rr - 1) * (current_level - lev);
                const int idx_max =
                    idx_min +
                    amrex::max<int>(0, rr * (current_level - lev) - 1);

                for (int idx = idx_min; idx <= idx_max; ++idx) {
                    const auto liquid_height =
                        vof_arr(ii, jj, k) * dz * mask_arr(i, j, k);
                    amrex::Gpu::Atomic::Add(
                        &uvof_sum[idx],
                        vel_arr(ii_v, jj_v, k, use_mac_fields ? 0 : idir) *
                            liquid_height);
                    amrex::Gpu::Atomic::Add(&vof_sum[idx], liquid_height);
                }
            });
        }
    }

    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, uvof_sum_d.begin(), uvof_sum_d.end(),
        uvof_sum_h.begin());
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, vof_sum_d.begin(), vof_sum_d.end(),
        dist_h.begin());

    amrex::ParallelDescriptor::ReduceRealSum(uvof_sum_h.data(), nline);
    amrex::ParallelDescriptor::ReduceRealSum(dist_h.data(), nline);

    for (int n = 0; n < nline; ++n) {
        avg_h[n] = (dist_h[n] > tiny) ? (uvof_sum_h[n] / dist_h[n]) : 0.0_rt;
    }
}

void Flather::compute_internal_z_averages()
{
    BL_PROFILE("kynema-sgf::Flather::compute_internal_z_averages");

    const int nlevels = m_repo.num_active_levels();
    const int nlevels_geom = static_cast<int>(m_mesh.Geom().size());
    if (m_xlo_uvof_avg.size() != nlevels_geom) {
        m_xlo_uvof_avg.resize(1, m_mesh.Geom());
        m_xhi_uvof_avg.resize(1, m_mesh.Geom());
        m_ylo_uvof_avg.resize(0, m_mesh.Geom());
        m_yhi_uvof_avg.resize(0, m_mesh.Geom());
        m_xlo_bnd_uvof_avg.resize(1, m_mesh.Geom());
        m_xhi_bnd_uvof_avg.resize(1, m_mesh.Geom());
        m_ylo_bnd_uvof_avg.resize(0, m_mesh.Geom());
        m_yhi_bnd_uvof_avg.resize(0, m_mesh.Geom());

        m_xlo_h_avg.resize(1, m_mesh.Geom());
        m_xhi_h_avg.resize(1, m_mesh.Geom());
        m_ylo_h_avg.resize(0, m_mesh.Geom());
        m_yhi_h_avg.resize(0, m_mesh.Geom());
        m_xlo_bnd_h_avg.resize(1, m_mesh.Geom());
        m_xhi_bnd_h_avg.resize(1, m_mesh.Geom());
        m_ylo_bnd_h_avg.resize(0, m_mesh.Geom());
        m_yhi_bnd_h_avg.resize(0, m_mesh.Geom());
    }

    for (int lev = 0; lev < nlevels; ++lev) {
        this->accumulate_boundary(
            lev, 0, true, m_xlo_uvof_avg, m_xlo_h_avg, false, FieldState::New);
        this->accumulate_boundary(
            lev, 0, false, m_xhi_uvof_avg, m_xhi_h_avg, false, FieldState::New);
        this->accumulate_boundary(
            lev, 1, true, m_ylo_uvof_avg, m_ylo_h_avg, false, FieldState::New);
        this->accumulate_boundary(
            lev, 1, false, m_yhi_uvof_avg, m_yhi_h_avg, false, FieldState::New);
    }

    m_xlo_uvof_avg.copy_host_to_device();
    m_xhi_uvof_avg.copy_host_to_device();
    m_ylo_uvof_avg.copy_host_to_device();
    m_yhi_uvof_avg.copy_host_to_device();

    m_xlo_h_avg.copy_host_to_device();
    m_xhi_h_avg.copy_host_to_device();
    m_ylo_h_avg.copy_host_to_device();
    m_yhi_h_avg.copy_host_to_device();
}

void Flather::compute_boundary_z_averages(
    int lev, FieldState fstate, bool use_mac_fields)
{
    BL_PROFILE("kynema-sgf::Flather::compute_boundary_z_averages");

    // accumulating boundaries needs to happen prior to applying fillpatch op

    this->accumulate_boundary(
        lev, 0, true, m_xlo_bnd_uvof_avg, m_xlo_bnd_h_avg, true, fstate,
        use_mac_fields);
    this->accumulate_boundary(
        lev, 0, false, m_xhi_bnd_uvof_avg, m_xhi_bnd_h_avg, true, fstate,
        use_mac_fields);
    this->accumulate_boundary(
        lev, 1, true, m_ylo_bnd_uvof_avg, m_ylo_bnd_h_avg, true, fstate,
        use_mac_fields);
    this->accumulate_boundary(
        lev, 1, false, m_yhi_bnd_uvof_avg, m_yhi_bnd_h_avg, true, fstate,
        use_mac_fields);

    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_xlo_bnd_uvof_avg.host_data(lev).begin(),
        m_xlo_bnd_uvof_avg.host_data(lev).end(),
        m_xlo_bnd_uvof_avg.device_data(lev).begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_xhi_bnd_uvof_avg.host_data(lev).begin(),
        m_xhi_bnd_uvof_avg.host_data(lev).end(),
        m_xhi_bnd_uvof_avg.device_data(lev).begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_ylo_bnd_uvof_avg.host_data(lev).begin(),
        m_ylo_bnd_uvof_avg.host_data(lev).end(),
        m_ylo_bnd_uvof_avg.device_data(lev).begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_yhi_bnd_uvof_avg.host_data(lev).begin(),
        m_yhi_bnd_uvof_avg.host_data(lev).end(),
        m_yhi_bnd_uvof_avg.device_data(lev).begin());

    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_xlo_bnd_h_avg.host_data(lev).begin(),
        m_xlo_bnd_h_avg.host_data(lev).end(),
        m_xlo_bnd_h_avg.device_data(lev).begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_xhi_bnd_h_avg.host_data(lev).begin(),
        m_xhi_bnd_h_avg.host_data(lev).end(),
        m_xhi_bnd_h_avg.device_data(lev).begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_ylo_bnd_h_avg.host_data(lev).begin(),
        m_ylo_bnd_h_avg.host_data(lev).end(),
        m_ylo_bnd_h_avg.device_data(lev).begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice, m_yhi_bnd_h_avg.host_data(lev).begin(),
        m_yhi_bnd_h_avg.host_data(lev).end(),
        m_yhi_bnd_h_avg.device_data(lev).begin());
}

void Flather::set_velocity(
    const int lev,
    const amrex::Real time,
    const Field& fld,
    amrex::MultiFab& mfab,
    const int dcomp,
    const int orig_comp) const
{
    BL_PROFILE("kynema-sgf::Flather::set_velocity");

    const auto& geom = m_mesh.Geom(lev);
    const auto& bctype = fld.bc_type();
    const int nghost = 1;
    const int numcomp = mfab.nComp();
    const auto& domain = geom.growPeriodicDomain(nghost);

    const auto fstate = time > 0.0_rt ? FieldState::Old : FieldState::New;
    const amrex::Real tiny = constants::TIGHT_TOL;
    const auto grav_z = -m_gravity[2];

    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if ((bctype[ori] != BC::mass_inflow) &&
            (bctype[ori] != BC::mass_inflow_outflow)) {
            continue;
        }

        const int idir = ori.coordDir();
        // Check if orientation aligns with supplied field and choose component
        bool skip_fill = false;
        int fcomp = 0;
        if (numcomp == 1) {
            // MAC velocity: only normal field is valid
            skip_fill = (orig_comp != idir);
            // Only a single component available
        } else {
            // Cell-centered velocity: field always valid
            fcomp = idir;
            // Select valid component
        }
        if (skip_fill) {
            continue;
        }

        const auto& dbx = ori.isLow() ? amrex::adjCellLo(domain, idir, nghost)
                                      : amrex::adjCellHi(domain, idir, nghost);
        const auto shift_to_interior =
            amrex::IntVect::TheDimensionVector(idir) * (ori.isLow() ? 1 : -1);
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
            const auto& ref_arr = fld.state(fstate)(lev)[mfi].const_array();

            const auto vof_arr = m_vof.state(fstate)(lev).const_array(mfi);
            const bool use_terrain = (m_terrain_blank != nullptr);
            const auto terrain_blank_arr =
                use_terrain ? (*m_terrain_blank)(lev).const_array(mfi)
                            : amrex::Array4<int const>();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const amrex::IntVect iv{i, j, k};
                // Need to account for shift to cell center, too!!
                const amrex::IntVect iv_adj = iv + shift_to_interior;

                amrex::Real boundary_val = arr(iv, fcomp);
                amrex::Real interior_val = arr(iv_adj, fcomp);
                amrex::Real boundary_h = 0.0_rt;
                amrex::Real interior_h = 0.0_rt;
                // Vectors at x boundaries extend in y direction
                // Vectors at y boundaries extend in x direction
                if (idir == 0) {
                    interior_val =
                        ori.isLow() ? xlo_uavg[iv_adj[1]] : xhi_uavg[iv_adj[1]];
                    interior_h =
                        ori.isLow() ? xlo_havg[iv_adj[1]] : xhi_havg[iv_adj[1]];
                    boundary_val =
                        ori.isLow() ? xlo_bnd_uavg[iv[1]] : xhi_bnd_uavg[iv[1]];
                    boundary_h =
                        ori.isLow() ? xlo_bnd_havg[iv[1]] : xhi_bnd_havg[iv[1]];
                } else {
                    interior_val =
                        ori.isLow() ? ylo_uavg[iv_adj[0]] : yhi_uavg[iv_adj[0]];
                    interior_h =
                        ori.isLow() ? ylo_havg[iv_adj[0]] : yhi_havg[iv_adj[0]];
                    boundary_val =
                        ori.isLow() ? ylo_bnd_uavg[iv[0]] : yhi_bnd_uavg[iv[0]];
                    boundary_h =
                        ori.isLow() ? ylo_bnd_havg[iv[0]] : yhi_bnd_havg[iv[0]];
                }

                // Set velocity to zero and skip for terrain
                if (use_terrain && terrain_blank_arr(iv_adj) != 0) {
                    arr(iv, fcomp) = 0.0_rt;
                    return;
                }
                // Check interior or boundary vof for liquid
                const amrex::Real interior_vof = vof_arr(iv_adj);
                const amrex::Real boundary_vof = vof_arr(iv);
                // Do nothing here if not liquid
                if ((interior_vof < tiny) && (boundary_vof < tiny)) {
                    return;
                }

                // Wave speed
                const amrex::Real c = std::sqrt(grav_z * interior_h);

                const auto Flather_val =
                    boundary_val + (ori.isLow() ? -1.0_rt : 1.0_rt) * c /
                                       boundary_h * (interior_h - boundary_h);

                const auto scaled_interior_vel =
                    ref_arr(iv_adj, idir) * (Flather_val / interior_val);

                // Only use if pointing outward or pulling liquid in
                // Zero velocity is fine either way
                const bool outflow = ori.isLow()
                                         ? scaled_interior_vel <= 0.0_rt
                                         : scaled_interior_vel >= 0.0_rt;
                const bool inflow_liq = !outflow && boundary_vof >= tiny;
                if (outflow || inflow_liq) {
                    arr(iv, fcomp) = scaled_interior_vel;
                }
            });
        }
    }
}

} // namespace kynema_sgf

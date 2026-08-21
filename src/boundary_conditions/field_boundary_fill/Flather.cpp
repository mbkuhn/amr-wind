#include "src/CFDSim.H"
#include "src/boundary_conditions/field_boundary_fill/Flather.H"
#include "src/boundary_conditions/field_boundary_fill/FillFlather.H"
#include "src/utilities/index_operations.H"
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
{
    amrex::ParmParse pp(identifier());

    pp.query("relaxation", m_relaxation);
    pp.query("target_weight", m_target_weight);
    pp.query("wind_speed", m_wind_speed);
    pp.query("wind_direction", m_wind_direction);
    pp.query("start_time", m_start_time);
    pp.query("stop_time", m_stop_time);
    pp.query("degrees_per_second", m_degrees_per_sec);

    update_target_velocity();
}

void Flather::post_init_actions()
{
    m_velocity.register_fill_patch_op<FillFlather>(m_mesh, m_time, *this);
}

void Flather::pre_advance_work()
{
#ifdef KYNEMA_SGF_USE_HELICS
    if (m_sim.helics().is_activated()) {
        m_wind_speed = m_sim.helics().m_inflow_wind_speed_to_kynema_sgf;
        m_wind_direction =
            -m_sim.helics().m_inflow_wind_direction_to_kynema_sgf + 270.0_rt;
        update_target_velocity();
        return;
    }
#endif

    if (m_time.current_time() > m_start_time &&
        m_time.current_time() < m_stop_time) {
        m_wind_direction -= m_degrees_per_sec * m_time.delta_t();
    }
    update_target_velocity();
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

    const amrex::Real relax = amrex::max(
        0.0_rt, amrex::min<amrex::Real>(1.0_rt, m_relaxation));
    const amrex::Real target_weight = amrex::max(
        0.0_rt, amrex::min<amrex::Real>(1.0_rt, m_target_weight));

    const amrex::Real tvx = m_uvec[0];
    const amrex::Real tvy = m_uvec[1];
    const amrex::Real tvz = m_uvec[2];

    const auto& geom = m_mesh.Geom(lev);
    const auto& bctype = fld.bc_type();
    const int nghost = 1;
    const auto& domain = geom.growPeriodicDomain(nghost);

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

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const amrex::IntVect iv{i, j, k};
                const amrex::IntVect iv_adj = iv + shift_to_interior;
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> target_vel =
                    {AMREX_D_DECL(tvx, tvy, tvz)};

                for (int n = 0; n < numcomp; ++n) {
                    const amrex::Real boundary_old = arr(iv, dcomp + n);
                    const amrex::Real interior_val = arr(iv_adj, dcomp + n);
                    const amrex::Real target_val = target_vel[orig_comp + n];

                    const amrex::Real desired =
                        ((1.0_rt - target_weight) * interior_val) +
                        (target_weight * target_val);

                    arr(iv, dcomp + n) =
                        ((1.0_rt - relax) * boundary_old) + (relax * desired);
                }
            });
        }
    }
}

void Flather::update_target_velocity()
{
    const amrex::Real wind_speed = m_wind_speed;
    const amrex::Real wind_direction = -m_wind_direction + 270.0_rt;
    const amrex::Real wind_direction_radian =
        kynema_sgf::utils::radians(wind_direction);

    m_uvec[0] = wind_speed * std::cos(wind_direction_radian);
    m_uvec[1] = wind_speed * std::sin(wind_direction_radian);
    m_uvec[2] = 0.0_rt;
}

} // namespace kynema_sgf

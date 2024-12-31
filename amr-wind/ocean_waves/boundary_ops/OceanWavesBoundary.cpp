#include "amr-wind/CFDSim.H"
#include "amr-wind/ocean_waves/boundary_ops/OceanWavesBoundary.H"
#include "amr-wind/ocean_waves/boundary_ops/OceanWavesFillInflow.H"
#include "amr-wind/utilities/index_operations.H"
#include "amr-wind/utilities/constants.H"
#include "amr-wind/core/Physics.H"
#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/physics/multiphase/MultiPhase.H"

namespace amr_wind {

OceanWavesBoundary::OceanWavesBoundary(CFDSim& sim)
    : m_time(sim.time())
    , m_repo(sim.repo())
    , m_mesh(sim.mesh())
    , m_ow_velocity(sim.repo().get_field("ow_velocity"))
    , m_ow_vof(sim.repo().get_field("ow_vof"))
{
    // Check for if boundary planes are being used; disable if so
    if (sim.physics_manager().contains("ABL")) {
        if (sim.physics_manager().get<amr_wind::ABL>().bndry_plane().mode() ==
            io_mode::input) {
            // Turn off ow_bndry; will rely on bndry_plane for fills
            m_activate_ow_bndry = false;
        }
        if (sim.physics_manager().get<amr_wind::ABL>().abl_mpl().is_active()) {
            amrex::Abort(
                "OceanWavesBoundary: not currently compatible with ABL MPL "
                "implementation.");
        }
    }
    // Get liquid density, will only be used if vof is present
    if (sim.physics_manager().contains("MultiPhase")) {
        m_rho1 = sim.physics_manager().get<amr_wind::MultiPhase>().rho1();
    }
}

void OceanWavesBoundary::post_init_actions()
{
    initialize_data();
    // Update boundary data (at n)
}

void OceanWavesBoundary::pre_advance_work()
{
    // Update boundary data for advection boundaries (nph for velocity, n for
    // vof)
}

void OceanWavesBoundary::pre_predictor_work()
{
    // Update boundary data for future fills (n+1 for velocity and vof)
}

void OceanWavesBoundary::initialize_data()
{
    BL_PROFILE("amr-wind::OceanWavesBoundary::initialize_data");
    if (m_activate_ow_bndry) {
        m_repo.get_field("velocity")
            .register_fill_patch_op<OceanWavesFillInflow>(
                m_mesh, m_time, *this);
        if (m_repo.field_exists("vof")) {
            m_repo.get_field("vof")
                .register_fill_patch_op<OceanWavesFillInflow>(
                    m_mesh, m_time, *this);
            m_repo.get_field("density")
                .register_fill_patch_op<OceanWavesFillInflow>(
                    m_mesh, m_time, *this);
        }
    }
}

void OceanWavesBoundary::set_velocity(
    const int lev,
    const amrex::Real /*time*/,
    const Field& fld,
    amrex::MultiFab& mfab,
    const int dcomp,
    const int orig_comp) const
{

    if (!m_activate_ow_bndry) {
        return;
    }

    BL_PROFILE("amr-wind::OceanWavesBoundary::set_velocity");

    const auto& geom = m_mesh.Geom(lev);
    const auto& bctype = fld.bc_type();
    const int nghost = 1;
    const auto& domain = geom.growPeriodicDomain(nghost);

    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if ((bctype[ori] != BC::mass_inflow) &&
            (bctype[ori] != BC::mass_inflow_outflow) &&
            (bctype[ori] != BC::wave_generation)) {
            continue;
        }

        const int idir = ori.coordDir();
        const auto& dbx = ori.isLow() ? amrex::adjCellLo(domain, idir, nghost)
                                      : amrex::adjCellHi(domain, idir, nghost);

        for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
            auto gbx = amrex::grow(mfi.validbox(), nghost);
            const auto& bx =
                utils::face_aware_boundary_box_intersection(gbx, dbx, ori);
            if (!bx.ok()) {
                continue;
            }

            const auto& targ_vof = m_ow_vof(lev).const_array(mfi);
            const auto& targ_arr = m_ow_velocity(lev).const_array(mfi);
            const auto& arr = mfab[mfi].array();
            const int numcomp = mfab.nComp();

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    for (int n = 0; n < numcomp; n++) {
                        if (targ_vof(i, j, k) > constants::LOOSE_TOL) {
                            arr(i, j, k, dcomp + n) =
                                targ_arr(i, j, k, orig_comp + n);
                        }
                    }
                });
        }
    }
}

void OceanWavesBoundary::set_vof(
    const int lev,
    const amrex::Real /*time*/,
    const Field& fld,
    amrex::MultiFab& mfab) const
{

    if (!m_activate_ow_bndry) {
        return;
    }

    BL_PROFILE("amr-wind::OceanWavesBoundary::set_vof");

    const auto& geom = m_mesh.Geom(lev);
    const auto& bctype = fld.bc_type();
    const int nghost = 1;
    const auto& domain = geom.growPeriodicDomain(nghost);

    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if ((bctype[ori] != BC::mass_inflow) &&
            (bctype[ori] != BC::mass_inflow_outflow) &&
            (bctype[ori] != BC::wave_generation)) {
            continue;
        }

        const int idir = ori.coordDir();
        const auto& dbx = ori.isLow() ? amrex::adjCellLo(domain, idir, nghost)
                                      : amrex::adjCellHi(domain, idir, nghost);

        for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
            auto gbx = amrex::grow(mfi.validbox(), nghost);
            const auto& bx =
                utils::face_aware_boundary_box_intersection(gbx, dbx, ori);
            if (!bx.ok()) {
                continue;
            }

            const auto& targ_arr = m_ow_vof(lev).const_array(mfi);
            const auto& arr = mfab[mfi].array();

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    arr(i, j, k) = targ_arr(i, j, k);
                });
        }
    }
}

void OceanWavesBoundary::set_density(
    const int lev,
    const amrex::Real /*time*/,
    const Field& fld,
    amrex::MultiFab& mfab) const
{

    if (!m_activate_ow_bndry || m_rho1 < 0.0) {
        return;
    }

    BL_PROFILE("amr-wind::OceanWavesBoundary::set_vof");

    const auto& geom = m_mesh.Geom(lev);
    const auto& bctype = fld.bc_type();
    const int nghost = 1;
    const amrex::Real rho1 = m_rho1;
    const auto& domain = geom.growPeriodicDomain(nghost);

    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if ((bctype[ori] != BC::mass_inflow) &&
            (bctype[ori] != BC::mass_inflow_outflow) &&
            (bctype[ori] != BC::wave_generation)) {
            continue;
        }

        const int idir = ori.coordDir();
        const auto& dbx = ori.isLow() ? amrex::adjCellLo(domain, idir, nghost)
                                      : amrex::adjCellHi(domain, idir, nghost);

        for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
            auto gbx = amrex::grow(mfi.validbox(), nghost);
            const auto& bx =
                utils::face_aware_boundary_box_intersection(gbx, dbx, ori);
            if (!bx.ok()) {
                continue;
            }

            const auto& targ_vof = m_ow_vof(lev).const_array(mfi);
            const auto& arr = mfab[mfi].array();

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    // Assume density is correct for gas phase only
                    arr(i, j, k) = rho1 * targ_vof(i, j, k) + arr(i, j, k);
                });
        }
    }
}

} // namespace amr_wind
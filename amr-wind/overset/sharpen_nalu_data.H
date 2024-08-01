#ifndef SHARPEN_NALU_DATA_H_
#define SHARPEN_NALU_DATA_H_

#include "amr-wind/CFDSim.H"
#include "amr-wind/physics/multiphase/MultiPhase.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/utilities/IOManager.H"
#include "amr-wind/overset/overset_ops_routines.H"

namespace amr_wind::overset {

amrex::Real calculate_pseudo_dt_flux(
    amrex::MultiFab& mf_fx,
    amrex::MultiFab& mf_fy,
    amrex::MultiFab& mf_fz,
    amrex::MultiFab& mf_vof,
    amrex::iMultiFab& mf_iblank,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
    amrex::Real tol)
{
    // Get the maximum flux magnitude, but just for vof fluxes
    const amrex::Real pdt_fx = amrex::ReduceMin(
        mf_fx, mf_vof, 0,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx, amrex::Array4<amrex::Real const> const& fx,
            amrex::Array4<amrex::Real const> const& vof) -> amrex::Real {
            amrex::Real pdt_fab = 1.0;
            amrex::Loop(bx, [=, &pdt_fab](int i, int j, int k) noexcept {
                amrex::Real pdt_lim = 1.0;
                if (fx(i, j, k, 0) > tol && vof(i, j, k) > tol) {
                    // VOF is removed from cell i
                    pdt_lim = vof(i, j, k) * dx[0] / fx(i, j, k, 0);
                } else if (fx(i, j, k, 0) < -tol && vof(i - 1, j, k) > tol) {
                    // VOF is removed from cell i-1
                    pdt_lim = vof(i - 1, j, k) * dx[0] / -fx(i, j, k, 0);
                }
                pdt_fab = amrex::min(pdt_fab, pdt_lim);
            });
            return pdt_fab;
        });
    const amrex::Real pdt_fy = amrex::ReduceMin(
        mf_fy, mf_vof, 0,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx, amrex::Array4<amrex::Real const> const& fy,
            amrex::Array4<amrex::Real const> const& vof) -> amrex::Real {
            amrex::Real pdt_fab = 1.0;
            amrex::Loop(bx, [=, &pdt_fab](int i, int j, int k) noexcept {
                amrex::Real pdt_lim = 1.0;
                if (fy(i, j, k, 0) > tol && vof(i, j, k) > tol) {
                    // VOF is removed from cell j
                    pdt_lim = vof(i, j, k) * dx[1] / fy(i, j, k, 0);
                } else if (fy(i, j, k, 0) < -tol && vof(i, j - 1, k) > tol) {
                    // VOF is removed from cell j-1
                    pdt_lim = vof(i, j - 1, k) * dx[1] / -fy(i, j, k, 0);
                }
                pdt_fab = amrex::min(pdt_fab, pdt_lim);
            });
            return pdt_fab;
        });
    // Pre-process z fluxes
    for (amrex::MFIter mfi(mf_fz); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();
        const amrex::Array4<const int>& iblank = mf_iblank.const_array(mfi);
        const amrex::Array4<amrex::Real>& fz = mf_fz.array(mfi);
        // Populate approximate signed distance function
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Ignore fluxes on overset boundaries
                if (iblank(i, j, k) * iblank(i, j, k - 1) == -1) {
                    fz(i, j, k, 8) = 0.0;
                }
            });
    }
    const amrex::Real pdt_fz = amrex::ReduceMin(
        mf_fz, mf_vof, 0,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx, amrex::Array4<amrex::Real const> const& fz,
            amrex::Array4<amrex::Real const> const& vof) -> amrex::Real {
            amrex::Real pdt_fab = 1.0;
            amrex::Loop(bx, [=, &pdt_fab](int i, int j, int k) noexcept {
                amrex::Real pdt_lim = 1.0;
                if (fz(i, j, k, 0) * fz(i, j, k, 8) > tol &&
                    vof(i, j, k) > tol) {
                    // VOF is removed from cell k
                    pdt_lim = vof(i, j, k) * dx[2] / fz(i, j, k, 0);
                } else if (
                    fz(i, j, k, 0) * fz(i, j, k, 8) < -tol &&
                    vof(i, j, k - 1) > tol) {
                    // VOF is removed from cell k-1
                    pdt_lim = vof(i, j, k - 1) * dx[2] / -fz(i, j, k, 0);
                }
                pdt_fab = amrex::min(pdt_fab, pdt_lim);
            });
            return pdt_fab;
        });
    const amrex::Real pdt = amrex::min(pdt_fx, amrex::min(pdt_fy, pdt_fz));
    return pdt;
}

void replace_gradp_hs(
    amrex::MultiFab& mf_gp,
    amrex::MultiFab& mf_density,
    amrex::MultiFab& mf_refdens,
    amrex::iMultiFab& mf_iblank,
    const amrex::Real grav_z,
    const bool is_pptb)
{
    for (amrex::MFIter mfi(mf_gp); mfi.isValid(); ++mfi) {
        const auto& gbx = mfi.growntilebox();
        const amrex::Array4<amrex::Real>& gp = mf_gp.array(mfi);
        const amrex::Array4<const amrex::Real>& rho =
            mf_density.const_array(mfi);
        const amrex::Array4<const amrex::Real>& rho0 =
            mf_refdens.const_array(mfi);
        const amrex::Array4<const int>& iblank = mf_iblank.const_array(mfi);
        amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Replace amr-wind vof values with originals
                if (iblank(i, j, k) == -1) {
                    const amrex::Real dfac =
                        is_pptb ? rho(i, j, k) - rho0(i, j, k) : rho(i, j, k);
                    gp(i, j, k, 0) = 0.;
                    gp(i, j, k, 1) = 0.;
                    gp(i, j, k, 2) = dfac * grav_z;
                }
            });
    }
}

void replace_gradp(
    amrex::MultiFab& mf_gp,
    amrex::MultiFab& mf_gp0,
    amrex::iMultiFab& mf_iblank,
    int ib_val)
{
    for (amrex::MFIter mfi(mf_gp); mfi.isValid(); ++mfi) {
        const auto& gbx = mfi.growntilebox();
        const amrex::Array4<amrex::Real>& gp = mf_gp.array(mfi);
        const amrex::Array4<const amrex::Real>& gp0 = mf_gp0.const_array(mfi);
        const amrex::Array4<const int>& iblank = mf_iblank.const_array(mfi);
        amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Replace amr-wind vof values with originals
                if (iblank(i, j, k) == ib_val) {
                    gp(i, j, k, 0) = gp0(i, j, k, 0);
                    gp(i, j, k, 1) = gp0(i, j, k, 1);
                    gp(i, j, k, 2) = gp0(i, j, k, 2);
                }
            });
    }
}

void replace_gradp(
    amrex::MultiFab& mf_gp,
    amrex::MultiFab& mf_gp0,
    amrex::iMultiFab& mf_iblank,
    amrex::iMultiFab& mf_iblank_node)
{
    for (amrex::MFIter mfi(mf_gp); mfi.isValid(); ++mfi) {
        const auto& gbx = mfi.growntilebox();
        const amrex::Array4<amrex::Real>& gp = mf_gp.array(mfi);
        const amrex::Array4<const amrex::Real>& gp0 = mf_gp0.const_array(mfi);
        const amrex::Array4<const int>& iblank = mf_iblank.const_array(mfi);
        const amrex::Array4<const int>& iblnod =
            mf_iblank_node.const_array(mfi);
        amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Replace amr-wind values with originals
                if (iblank(i, j, k) == -1) {
                    bool check = iblnod(i, j, k) == -1;
                    check = check && iblnod(i + 1, j, k) == -1;
                    check = check && iblnod(i, j + 1, k) == -1;
                    check = check && iblnod(i, j, k + 1) == -1;
                    check = check && iblnod(i + 1, j + 1, k) == -1;
                    check = check && iblnod(i, j + 1, k - 1) == -1;
                    check = check && iblnod(i + 1, j, k + 1) == -1;
                    check = check && iblnod(i + 1, j + 1, k + 1) == -1;
                    if (!check) {
                        gp(i, j, k, 0) = gp0(i, j, k, 0);
                        gp(i, j, k, 1) = gp0(i, j, k, 1);
                        gp(i, j, k, 2) = gp0(i, j, k, 2);
                    }
                } else {
                    gp(i, j, k, 0) = gp0(i, j, k, 0);
                    gp(i, j, k, 1) = gp0(i, j, k, 1);
                    gp(i, j, k, 2) = gp0(i, j, k, 2);
                }
            });
    }
}

void apply_pressure_gradient(
    amrex::MultiFab& mf_vel,
    amrex::MultiFab& mf_density,
    amrex::MultiFab& mf_gp,
    const amrex::Real scaling_factor)
{
    for (amrex::MFIter mfi(mf_gp); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();
        const amrex::Array4<amrex::Real>& vel = mf_vel.array(mfi);
        const amrex::Array4<const amrex::Real>& rho =
            mf_density.const_array(mfi);
        const amrex::Array4<const amrex::Real>& gp = mf_gp.const_array(mfi);
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real soverrho = scaling_factor / rho(i, j, k);
                vel(i, j, k, 0) -= gp(i, j, k, 0) * soverrho;
                vel(i, j, k, 1) -= gp(i, j, k, 1) * soverrho;
                vel(i, j, k, 2) -= gp(i, j, k, 2) * soverrho;
            });
    }
}

void SharpenNaluDataDiscrete(
    CFDSim& sim,
    const int nit,
    const amrex::Real tol,
    const int ncconv,
    const amrex::Real rlscale,
    const amrex::Real margin,
    const amrex::Real proc_tgvof_tol,
    const bool sh_pressure)
{
    auto& repo = sim.repo();
    auto nlevels = repo.num_active_levels();
    auto geom = sim.mesh().Geom();

    // Check for vof variable -- sharpening only required with VOF
    bool yesVOF = repo.field_exists("vof");
    if (!yesVOF) {
        return;
    }

    // Get phase densities
    auto& mphase = sim.physics_manager().get<MultiPhase>();
    const amrex::Real rhol = mphase.rho1();
    const amrex::Real rhog = mphase.rho2();

    // Get blanking for cells
    auto& iblank_cell = repo.get_int_field("iblank_cell");
    auto& iblank_node = repo.get_int_field("iblank_node");

    // Get fields that will be modified
    auto& vof = repo.get_field("vof");
    auto& levelset = repo.get_field("levelset");
    auto& rho = repo.get_field("density");
    auto& velocity = repo.get_field("velocity");
    auto& gp_noghost = repo.get_field("gp");
    auto& p = repo.get_field("p");

    // Create scratch fields for fluxes - components are vof, density, and
    // 3 of velocity, 3 of pressure gradient, 1 addl component for p_src flag
    auto flux_x = repo.create_scratch_field(9, 1, amr_wind::FieldLoc::XFACE);
    auto flux_y = repo.create_scratch_field(9, 1, amr_wind::FieldLoc::YFACE);
    auto flux_z = repo.create_scratch_field(9, 1, amr_wind::FieldLoc::ZFACE);
    // Create scratch field for pressure source term
    auto p_src = repo.create_scratch_field(1, 0, amr_wind::FieldLoc::NODE);
    // Create scratch fields for normal vector and target vof field
    auto normal_vec = repo.create_scratch_field(3, vof.num_grow()[0] - 1);
    auto target_vof = repo.create_scratch_field(1, vof.num_grow()[0]);
    // Create scratch field for pressure gradient that has ghost cells
    // Sharpening fluxes (at faces) have 1 ghost, requiring fields to have >= 2
    auto gp_scr = repo.create_scratch_field(3, 2);
    auto& gp = *gp_scr;

    // Give initial max possible value of pseudo-velocity scale
    auto dx_lev0 = (geom[0]).CellSizeArray();
    const amrex::Real max_pvscale =
        std::min(std::min(dx_lev0[0], dx_lev0[1]), dx_lev0[2]);
    amrex::Real pvscale = max_pvscale;

    // Prep things that do not change with iterations
    for (int lev = 0; lev < nlevels; ++lev) {

        // Thickness used here is user parameter, whatever works best
        auto dx = (geom[lev]).CellSizeArray();
        const amrex::Real i_th = rlscale * std::cbrt(dx[0] * dx[1] * dx[2]);

        // Populate approximate signed distance function
        amr_wind::overset_ops::populate_psi(
            levelset(lev), vof(lev), i_th, 1e-12);

        // Populate gp scratch field
        gp(lev).setVal(0.0); // for external boundaries
        amrex::MultiFab::Copy(gp(lev), gp_noghost(lev), 0, 0, 3, 0); // nonghost
        gp(lev).FillBoundary(sim.mesh().Geom()[lev].periodicity());  // internal

        // Get pseudo-velocity scale, proportional to smallest dx in iblank
        const amrex::Real pvscale_lev =
            amr_wind::overset_ops::calculate_pseudo_velocity_scale(
                iblank_cell(lev), dx, max_pvscale);
        pvscale = std::min(pvscale, pvscale_lev);
    }
    amrex::Gpu::synchronize();
    amrex::ParallelDescriptor::ReduceRealMin(pvscale);

    // Convert levelset to vof to get target_vof
    mphase.levelset2vof(iblank_cell, *target_vof);

    // Process target vof for tiny margins from single-phase
    for (int lev = 0; lev < nlevels; ++lev) {
        // A tolerance of 0 should do nothing
        amr_wind::overset_ops::process_vof((*target_vof)(lev), proc_tgvof_tol);
    }
    amrex::Gpu::synchronize();

    // Replace vof with original values in amr domain
    for (int lev = 0; lev < nlevels; ++lev) {
        amr_wind::overset_ops::harmonize_vof(
            (*target_vof)(lev), vof(lev), iblank_cell(lev));
    }
    amrex::Gpu::synchronize();

    // Put fluxes in vector for averaging down during iterations
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> fluxes(
        repo.num_active_levels());
    for (int lev = 0; lev < nlevels; ++lev) {
        fluxes[lev][0] = &(*flux_x)(lev);
        fluxes[lev][1] = &(*flux_y)(lev);
        fluxes[lev][2] = &(*flux_z)(lev);
    }

    // Pseudo-time loop
    amrex::Real err = 100.0 * tol;
    int n = 0;
    while (n < nit && err > tol) {
        // Increment step counter
        ++n;
        // Determine if convergence error is calculated this step
        bool cconv = n % ncconv == 0;
        // Zero error if being calculated this step
        err = cconv ? 0.0 : err;

        // Maximum possible value of pseudo time factor (dtau)
        amrex::Real ptfac = 1.0;

        // Maximum pseudoCFL, 0.5 seems to work well
        const amrex::Real pCFL = 0.5;

        for (int lev = 0; lev < nlevels; ++lev) {
            // Populate normal vector
            amr_wind::overset_ops::populate_normal_vector(
                (*normal_vec)(lev), vof(lev), iblank_cell(lev));

            // Sharpening fluxes for vof, density, and momentum
            amr_wind::overset_ops::populate_sharpen_fluxes(
                (*flux_x)(lev), (*flux_y)(lev), (*flux_z)(lev), vof(lev),
                (*target_vof)(lev), (*normal_vec)(lev), velocity(lev), gp(lev),
                rho(lev), pvscale, margin, rhol, rhog);

            // Process fluxes and get pressure source term
            amr_wind::overset_ops::process_fluxes_calc_src(
                (*flux_x)(lev), (*flux_y)(lev), (*flux_z)(lev), (*p_src)(lev),
                vof(lev), iblank_cell(lev));

            // Measure convergence to determine if loop can stop
            if (cconv) {
                // Update error at specified interval of steps
                const amrex::Real err_lev =
                    amr_wind::overset_ops::measure_convergence(
                        (*flux_x)(lev), (*flux_y)(lev), (*flux_z)(lev)) /
                    pvscale;
                err = amrex::max(err, err_lev);
            }
        }

        // Average down fluxes across levels for consistency
        for (int lev = nlevels - 1; lev > 0; --lev) {
            amrex::IntVect rr =
                geom[lev].Domain().size() / geom[lev - 1].Domain().size();
            amrex::average_down_faces(
                GetArrOfConstPtrs(fluxes[lev]), fluxes[lev - 1], rr,
                geom[lev - 1]);
        }

        // Get pseudo dt (dtau)
        for (int lev = 0; lev < nlevels; ++lev) {
            // Compare vof fluxes to vof in source cells
            auto dx = (geom[lev]).CellSizeArray();
            // Convergence tolerance determines what size of fluxes matter
            const amrex::Real ptfac_lev = calculate_pseudo_dt_flux(
                (*flux_x)(lev), (*flux_y)(lev), (*flux_z)(lev), vof(lev),
                iblank_cell(lev), dx, tol);
            ptfac = amrex::min(ptfac, ptfac_lev);
        }
        amrex::ParallelDescriptor::ReduceRealMin(ptfac);

        // Conform pseudo dt (dtau) to pseudo CFL
        ptfac = pCFL * ptfac;

        // Apply fluxes
        for (int lev = 0; lev < nlevels; ++lev) {
            auto dx = (geom[lev]).CellSizeArray();

            amr_wind::overset_ops::apply_fluxes(
                (*flux_x)(lev), (*flux_y)(lev), (*flux_z)(lev), (*p_src)(lev),
                vof(lev), rho(lev), velocity(lev), gp(lev), p(lev), dx, ptfac,
                1e-12);

            // Update ghost cells
            vof(lev).FillBoundary(sim.mesh().Geom()[lev].periodicity());
            velocity(lev).FillBoundary(sim.mesh().Geom()[lev].periodicity());
            gp(lev).FillBoundary(sim.mesh().Geom()[lev].periodicity());
        }

        // Update density according to vof (fillpatch built in)
        mphase.set_density_via_vof();

        // Copy gp so it can be seen
        amr_wind::field_ops::copy(
            gp_noghost, gp, 0, 0, gp.num_comp(), gp_noghost.num_grow());
        // Output for debugging
        // sim.io_manager().write_plot_file();
        // sim.time().increment_timestep();

        // Ensure that err is same across processors
        if (cconv) {
            amrex::ParallelDescriptor::ReduceRealMax(err);
        }

        amrex::Print() << "sharpen step " << n << " " << err << " " << tol
                       << std::endl;
    }

    // Purely for debugging via visualization, should be removed later
    // Currently set up to overwrite the levelset field (not used as time
    // evolves) with the post-sharpening velocity magnitude
    for (int lev = 0; lev < nlevels; ++lev) {
        amr_wind::overset_ops::equate_field(levelset(lev), velocity(lev));
    }

    // Copy to old
    amr_wind::field_ops::copy(
        vof.state(amr_wind::FieldState::Old), vof, 0, 0, vof.num_comp(),
        vof.num_grow());
    amr_wind::field_ops::copy(
        rho.state(amr_wind::FieldState::Old), rho, 0, 0, rho.num_comp(),
        rho.num_grow());
    amr_wind::field_ops::copy(
        velocity.state(amr_wind::FieldState::Old), velocity, 0, 0,
        velocity.num_comp(), velocity.num_grow());

    // Fillpatch for pressure to make sure pressure stencil has all points
    p.fillpatch(sim.time().current_time());
}

void ReplaceMaskedGradP(CFDSim& sim)
{
    auto& repo = sim.repo();
    auto nlevels = repo.num_active_levels();
    auto geom = sim.mesh().Geom();

    // Get gravity
    auto& mphase = sim.physics_manager().get<MultiPhase>();
    const amrex::Real grav_z = mphase.gravity()[2];
    const bool is_pptb = mphase.perturb_pressure();

    // Get blanking for cells
    auto& iblank_cell = repo.get_int_field("iblank_cell");

    // Get fields that will be modified or used
    Field* rho0{nullptr};
    auto& rho = repo.get_field("density");
    auto& gp = repo.get_field("gp");
    if (is_pptb) {
        rho0 = &(sim.repo().get_field("reference_density"));
    } else {
        // Point to existing field, won't be used
        rho0 = &rho;
    }

    // Replace initial gp with best guess (hydrostatic)
    for (int lev = 0; lev < nlevels; ++lev) {
        replace_gradp_hs(
            gp(lev), rho(lev), (*rho0)(lev), iblank_cell(lev), grav_z, is_pptb);
    }
}

void ReapplyModifiedGradP(CFDSim& sim)
{
    auto& repo = sim.repo();
    auto nlevels = repo.num_active_levels();
    auto geom = sim.mesh().Geom();

    // Get gravity
    auto& mphase = sim.physics_manager().get<MultiPhase>();
    const amrex::Real grav_z = mphase.gravity()[2];
    const bool is_pptb = mphase.perturb_pressure();

    // Get timestep
    const amrex::Real dt = sim.time().delta_t();

    // Get blanking for cells
    auto& iblank_cell = repo.get_int_field("iblank_cell");

    // Get fields that will be modified or used
    Field* rho0{nullptr};
    auto& vel = repo.get_field("velocity");
    auto& rho = repo.get_field("density");
    auto& gp = repo.get_field("gp");
    if (is_pptb) {
        rho0 = &(sim.repo().get_field("reference_density"));
    } else {
        // Point to existing field, won't be used
        rho0 = &rho;
    }

    // For iblanked cells, replace gp with original gp, to get original vel
    for (int lev = 0; lev < nlevels; ++lev) {
        // Remove pressure gradient term
        apply_pressure_gradient(vel(lev), rho(lev), gp(lev), -dt);
        // Modify pressure gradient
        replace_gradp_hs(
            gp(lev), rho(lev), (*rho0)(lev), iblank_cell(lev), grav_z, is_pptb);
        // Reapply pressure gradient term
        apply_pressure_gradient(vel(lev), rho(lev), gp(lev), dt);
    }
}

void CopyGradP(ScratchField& gp_copy, Field& gp, const int nlevels)
{
    // For iblanked cells, replace gp with original gp, to get original vel
    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::MultiFab::Copy(
            gp_copy(lev), gp(lev), 0, 0, gp(lev).nComp(), gp(lev).nGrow());
    }
}

void ReapplyOversetGradP(ScratchField& gp_copy, CFDSim& sim)
{
    auto& repo = sim.repo();
    auto nlevels = repo.num_active_levels();

    // Get timestep
    const amrex::Real dt = sim.time().delta_t();

    // Get blanking for cells
    auto& iblank_cell = repo.get_int_field("iblank_cell");

    // Get fields that will be modified or used
    auto& vel = repo.get_field("velocity");
    auto& rho = repo.get_field("density");
    auto& gp = repo.get_field("gp");

    // For iblanked cells, replace gp with original gp, to get original vel
    for (int lev = 0; lev < nlevels; ++lev) {
        // Remove pressure gradient term
        apply_pressure_gradient(vel(lev), rho(lev), gp(lev), -dt);
        // Modify pressure gradient, replace blanked version
        replace_gradp(gp(lev), gp_copy(lev), iblank_cell(lev), -1);
        // Reapply pressure gradient term
        apply_pressure_gradient(vel(lev), rho(lev), gp(lev), dt);
    }
}

void ReplaceUnMaskedGradP(
    Field& gp, ScratchField& gp_copy, IntField& iblank_cell)
{
    // Replace gp with copy from before exchange
    for (int lev = 0; lev < gp.repo().num_active_levels(); ++lev) {
        replace_gradp(gp(lev), gp_copy(lev), iblank_cell(lev), 1);
    }
}

void ReplaceUnMaskedGradP(
    Field& gp,
    ScratchField& gp_copy,
    IntField& iblank_cell,
    IntField& iblank_node)
{
    // Replace gp with copy from before exchange
    for (int lev = 0; lev < gp.repo().num_active_levels(); ++lev) {
        replace_gradp(
            gp(lev), gp_copy(lev), iblank_cell(lev), iblank_node(lev));
    }
}

} // namespace amr_wind::overset

#endif
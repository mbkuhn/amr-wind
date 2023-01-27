#ifndef PLANE_RECONSTRUCT_H_
#define PLANE_RECONSTRUCT_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFabUtil.H>
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/core/FieldRepo.H"

namespace amr_wind {
namespace multiphase {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void fit_plane(
    const int i,
    const int j,
    const int k,
    const int method,
    amrex::Array4<amrex::Real const> const& volfrac,
    amrex::Array4<amrex::Real> const& iplane)
{

    amrex::Real mx, my, mz, alpha;
    switch (method) {
    case 2:
        swartz_normal(i, j, k, volfrac, iplane, mx, my, mz);
        break;
    case 1:
        swartz_onestep_normal(i, j, k, volfrac, mx, my, mz);
        break;
    case 0:
    default:
        mixed_youngs_central_normal(i, j, k, volfrac, mx, my, mz);
        break;
    }

    amrex::Real invx = 1.0;
    amrex::Real invy = 1.0;
    amrex::Real invz = 1.0;

    if (mx < 0.0) {
        mx = -mx;
        invx = -1.0;
    }
    if (my < 0.0) {
        my = -my;
        invy = -1.0;
    }
    if (mz < 0.0) {
        mz = -mz;
        invz = -1.0;
    }

    amrex::Real mm2 = mx + my + mz;
    mx = mx / mm2;
    my = my / mm2;
    mz = mz / mm2;

    alpha = volume_intercept(mx, my, mz, volfrac(i, j, k));

    // Back to the original plane
    mx = invx * mx;
    my = invy * my;
    mz = invz * mz;
    alpha = alpha + amrex::min<amrex::Real>(0.0, mx) +
            amrex::min<amrex::Real>(0.0, my) + amrex::min<amrex::Real>(0.0, mz);

    // Store answer
    iplane(i, j, k, 0) = mx;
    iplane(i, j, k, 1) = my;
    iplane(i, j, k, 2) = mz;
    iplane(i, j, k, 3) = alpha;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void fit_plane(
    const int i,
    const int j,
    const int k,
    amrex::Array4<amrex::Real const> const& volfrac,
    amrex::Array4<amrex::Real> const& iplane)
{
    fit_plane(i, j, k, 0, volfrac, iplane);
}

static void reconstruct_planes_onestep(
    int nlevels, int method, Field& vof, Field& iplane)
{

    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::MFItInfo mfi_info;
        if (amrex::Gpu::notInLaunchRegion()) {
            mfi_info.EnableTiling(amrex::IntVect(1024, 1024, 1024))
                .SetDynamic(true);
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(vof(lev), mfi_info); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& vof_arr = vof(lev).const_array(mfi);
            const auto& pl_arr = iplane(lev).array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    // Sum fluxes from this stage of advection
                    fit_plane(i, j, k, method, vof_arr, pl_arr);
                });
        }
    }

    // Average down vof and communicate
    iplane.fillpatch(0.0);
}

static void reconstruct_planes_swartz(int nlevels, Field& vof, Field& iplane)
{
    constexpr amrex::Real tol = 1.0e-6;
    constexpr int maxiter = 5;

    // Initialize with MYC normals
    reconstruct_planes_onestep(nlevels, 0, vof, iplane);

    // While loop limited by iteration count
    amrex::Real pl_res = 1.0;
    int count = 0;
    while (pl_res > tol && count < maxiter) {

        pl_res = 0.0;

        for (int lev = 0; lev < nlevels; ++lev) {
            amrex::MFItInfo mfi_info;
            if (amrex::Gpu::notInLaunchRegion()) {
                mfi_info.EnableTiling(amrex::IntVect(1024, 1024, 1024))
                    .SetDynamic(true);
            }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            // Create copy of current normal information
            amrex::MultiFab oldplane(
                iplane(lev).boxArray(), iplane(lev).DistributionMap(), 3, 0);
            amrex::MultiFab::Copy(oldplane, iplane(lev), 0, 0, 3, 0);

            for (amrex::MFIter mfi(vof(lev), mfi_info); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                const auto& vof_arr = vof(lev).const_array(mfi);
                const auto& pl_arr = iplane(lev).array(mfi);

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        // Sum fluxes from this stage of advection
                        fit_plane(i, j, k, 2, vof_arr, pl_arr);
                    });
            }

            // Get dot product of new plane with old plane
            amrex::MultiFab::Multiply(oldplane, iplane(lev), 0, 0, 3, 0);

            // Reduce product and sum with other levels
            pl_res += amrex::ReduceSum(
                oldplane, 0,
                [=] AMREX_GPU_HOST_DEVICE(
                    amrex::Box const& bx,
                    amrex::Array4<amrex::Real const> const& pl_arr)
                    -> amrex::Real {
                    amrex::Real pl_res_fab = 0.0;

                    amrex::Loop(
                        bx, [=, &pl_res_fab](int i, int j, int k) noexcept {
                            pl_res_fab +=
                                (1.0 -
                                 (pl_arr(i, j, k, 0) + pl_arr(i, j, k, 1) +
                                  pl_arr(i, j, k, 2)));
                        });
                    return pl_res_fab;
                });
        }

        // Sum residual and increment counter
        amrex::ParallelDescriptor::ReduceRealSum(pl_res);
        ++count;

        // Average down vof and communicate
        iplane.fillpatch(0.0);
    }
}

static void reconstruct_planes(int method, Field& vof, Field& iplane)
{
    int nlevels = vof.repo().num_active_levels();
    switch (method) {
    case 2:
        reconstruct_planes_swartz(nlevels, vof, iplane);
        break;
    case 1:
    case 0:
    default:
        reconstruct_planes_onestep(nlevels, method, vof, iplane);
        break;
    }
}

} // namespace multiphase
} // namespace amr_wind
#endif
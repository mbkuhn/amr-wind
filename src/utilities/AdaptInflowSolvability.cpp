#include "src/utilities/AdaptInflowSolvability.H"

#include "AMReX_BC_TYPES.H"
#include "AMReX_GpuReduce.H"
#include "AMReX_iMultiFab.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Orientation.H"
#include "AMReX_ParReduce.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_MultiFabUtil.H"

#include <cmath>

using namespace amrex;

namespace kynema_sgf {

namespace {

// Mask values used to tag each boundary cell category
constexpr int AI_INFLOW_TAG  = -1;
constexpr int AI_OUTFLOW_TAG = +1;
constexpr int AI_PASSIVE_TAG = +2;

constexpr Real small_vel = 1.e-8_rt;

/** Tag each adapt_inflow boundary cell as inflow, outflow, or passive.
 *
 *  For a low boundary, the categorisation is:
 *    inflow:  vel_boundary > 0        (points into domain)
 *    outflow: vel_boundary <= 0  AND  vel_interior < 0
 *    passive: vel_boundary <= 0  AND  vel_interior >= 0
 *
 *  The sign conventions are reversed for high boundaries.
 */
void set_adapt_inflow_masks(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Array<iMultiFab, AMREX_SPACEDIM>& level_masks,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const bool corners)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) { continue; }

        const int dir = ori.coordDir();
        const bool oriIsLow  = ori.isLow();
        const bool oriIsHigh = ori.isHigh();

        const auto& vel_mf = vels_vec[lev][dir];
        auto& mask         = masks[dir];
        const auto& level_mask = level_masks[dir];

        const IndexType::CellIndex dir_idx_type =
            (vel_mf->ixType()).ixType(dir);

        // boundary plane index in the normal direction
        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(dir) - 1
                            : domain.smallEnd(dir);
        const int dhi = domain.bigEnd(dir) + 1;
        const int bndry = oriIsLow ? dlo : dhi;

        // unit step from boundary toward domain interior
        const int di = (dir == 0) ? 1 : 0;
        const int dj = (dir == 1) ? 1 : 0;
        const int dk = (dir == 2) ? 1 : 0;

        for (MFIter mfi(*vel_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box box = mfi.tilebox();
            const IntVect ungrown_sm = box.smallEnd();
            const IntVect ungrown_bg = box.bigEnd();

            if (dir_idx_type == IndexType::CellIndex::CELL) {
                box.grow(dir, 1);
            }
            if (corners) {
                const int t1 = (dir + 1) % AMREX_SPACEDIM;
                if (box.smallEnd(t1) == domain.smallEnd(t1)) { box.growLo(t1, 1); }
                if (box.bigEnd(t1)   == domain.bigEnd(t1))   { box.growHi(t1, 1); }
#if (AMREX_SPACEDIM == 3)
                const int t2 = (dir + 2) % AMREX_SPACEDIM;
                if (box.smallEnd(t2) == domain.smallEnd(t2)) { box.growLo(t2, 1); }
                if (box.bigEnd(t2)   == domain.bigEnd(t2))   { box.growHi(t2, 1); }
#endif
            }

            const bool at_lo_bndry = oriIsLow  && (box.smallEnd(dir) == dlo);
            const bool at_hi_bndry = oriIsHigh && (box.bigEnd(dir)   == dhi);
            if (!at_lo_bndry && !at_hi_bndry) { continue; }

            Box box2d(box); box2d.setRange(dir, bndry);

            const auto vel_arr        = vel_mf->array(mfi);
            auto       mask_arr       = mask.array(mfi);
            const auto level_mask_arr = level_mask.const_array(mfi);

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const IntVect iv{AMREX_D_DECL(i, j, k)};
                const IntVect lm_idx =
                    amrex::max(amrex::min(iv, ungrown_bg), ungrown_sm);
                if (level_mask_arr(lm_idx) != 1) { return; }

                const Real vb = vel_arr(i, j, k);
                const Real vi = oriIsLow ? vel_arr(i + di, j + dj, k + dk)
                                         : vel_arr(i - di, j - dj, k - dk);

                if ((oriIsLow && vb > 0) || (oriIsHigh && vb < 0)) {
                    mask_arr(i, j, k) = AI_INFLOW_TAG;
                } else if ((oriIsLow && vi < 0) || (oriIsHigh && vi > 0)) {
                    mask_arr(i, j, k) = AI_OUTFLOW_TAG;
                } else {
                    mask_arr(i, j, k) = AI_PASSIVE_TAG;
                }
            });
        }
    }
}

/** Accumulate influx, outflux, and total passive face area from the masks. */
void compute_adapt_inflow_fluxes(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Real* a_dx,
    Real& influx,
    Real& outflux,
    Real& passive_area,
    const bool corners)
{
    influx = 0.0; outflux = 0.0; passive_area = 0.0;

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        const Real ds = a_dx[(idim + 1) % AMREX_SPACEDIM]
                      * a_dx[(idim + 2) % AMREX_SPACEDIM];

        const auto& vel_mf = vels_vec[lev][idim];

        // grow vector: 1 in normal dir for cell-centered, 0 for face-centered
        IndexType idx_type = vel_mf->ixType();
        idx_type.flip(idim);
        IntVect ngrow = idx_type.ixType();
        if (corners) {
            ngrow[(idim + 1) % AMREX_SPACEDIM] = 1;
#if (AMREX_SPACEDIM == 3)
            ngrow[(idim + 2) % AMREX_SPACEDIM] = 1;
#endif
        }

        const auto& mask = masks[idim];
        const auto vel_ma  = vel_mf->const_arrays();
        const auto mask_ma = mask.const_arrays();

        influx += ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                if (mask_ma[box_no](i, j, k) == AI_INFLOW_TAG) {
                    return {std::abs(vel_ma[box_no](i, j, k))};
                }
                return {0.0_rt};
            });

        outflux += ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                if (mask_ma[box_no](i, j, k) == AI_OUTFLOW_TAG) {
                    return {std::abs(vel_ma[box_no](i, j, k))};
                }
                return {0.0_rt};
            });

        passive_area += ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                if (mask_ma[box_no](i, j, k) == AI_PASSIVE_TAG) {
                    return {1.0_rt};
                }
                return {0.0_rt};
            });
    }

    ParallelDescriptor::ReduceRealSum(influx);
    ParallelDescriptor::ReduceRealSum(outflux);
    ParallelDescriptor::ReduceRealSum(passive_area);
}

/** Set boundary velocity for passive cells to achieve mass balance.
 *
 *  Passive cells are identified by the same two-velocity check used in
 *  set_adapt_inflow_masks rather than reading the mask iMultiFab, so that
 *  this function can be called in a second pass after all fluxes are known.
 */
void apply_passive_inflow(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const Real v_passive,
    const bool corners)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) { continue; }

        const int dir = ori.coordDir();
        const bool oriIsLow  = ori.isLow();
        const bool oriIsHigh = ori.isHigh();

        const auto& vel_mf = vels_vec[lev][dir];

        const IndexType::CellIndex dir_idx_type =
            (vel_mf->ixType()).ixType(dir);

        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(dir) - 1
                            : domain.smallEnd(dir);
        const int dhi  = domain.bigEnd(dir) + 1;
        const int bndry = oriIsLow ? dlo : dhi;

        const int di = (dir == 0) ? 1 : 0;
        const int dj = (dir == 1) ? 1 : 0;
        const int dk = (dir == 2) ? 1 : 0;

        // inward speed: positive for low boundary, negative for high
        const Real v_inward = oriIsLow ? v_passive : -v_passive;

        for (MFIter mfi(*vel_mf, false); mfi.isValid(); ++mfi) {
            Box box = mfi.validbox();
            if (dir_idx_type == IndexType::CellIndex::CELL) {
                box.grow(dir, 1);
            }
            if (corners) {
                box.grow((dir + 1) % AMREX_SPACEDIM, 1);
#if (AMREX_SPACEDIM == 3)
                box.grow((dir + 2) % AMREX_SPACEDIM, 1);
#endif
            }

            if ((oriIsLow  && (box.smallEnd(dir) != dlo)) ||
                (oriIsHigh && (box.bigEnd(dir)   != dhi))) {
                continue;
            }

            Box box2d(box); box2d.setRange(dir, bndry);
            auto vel_arr = vel_mf->array(mfi);

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const Real vb = vel_arr(i, j, k);
                const Real vi = oriIsLow ? vel_arr(i + di, j + dj, k + dk)
                                         : vel_arr(i - di, j - dj, k - dk);

                // passive: boundary vel not inward AND interior vel not outward
                const bool is_passive =
                    (oriIsLow  && vb <= 0 && vi >= 0) ||
                    (oriIsHigh && vb >= 0 && vi <= 0);

                if (is_passive) { vel_arr(i, j, k) = v_inward; }
            });
        }
    }
}

} // file-local namespace

void enforceAdaptInflowSolvability(
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Vector<Geometry>& geom,
    const bool include_bndry_corners)
{
    bool has_adapt_inflow = false;
    for (OrientationIter oit; oit != nullptr; ++oit) {
        if (bc_types[oit()] == BC::adapt_inflow) {
            has_adapt_inflow = true;
            break;
        }
    }
    if (!has_adapt_inflow) { return; }

    const int nlevs = static_cast<int>(vels_vec.size());
    Real influx = 0.0, outflux = 0.0, passive_area = 0.0;

    for (int lev = 0; lev < nlevs; ++lev) {
        const Box domain = geom[lev].Domain();

        Array<iMultiFab, AMREX_SPACEDIM> masks;
        Array<iMultiFab, AMREX_SPACEDIM> level_masks;

        IntVect rr;
        if (lev < nlevs - 1) {
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
                rr[idim] = static_cast<int>(std::round(
                    geom[lev].CellSize(idim) / geom[lev + 1].CellSize(idim)));
            }
        }

        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            const auto& vel_mf = vels_vec[lev][idim];

            IndexType idx_type = vel_mf->ixType();
            idx_type.flip(idim);
            IntVect ngrow = idx_type.ixType();
            if (include_bndry_corners) {
                ngrow[(idim + 1) % AMREX_SPACEDIM] = 1;
#if (AMREX_SPACEDIM == 3)
                ngrow[(idim + 2) % AMREX_SPACEDIM] = 1;
#endif
            }

            masks[idim].define(
                vel_mf->boxArray(), vel_mf->DistributionMap(), 1, ngrow);
            masks[idim].setVal(0);

            if (lev < nlevs - 1) {
                level_masks[idim] = makeFineMask(
                    vel_mf->boxArray(), vel_mf->DistributionMap(),
                    vels_vec[lev + 1][idim]->boxArray(), rr, 1, 0);
            } else {
                level_masks[idim].define(
                    vel_mf->boxArray(), vel_mf->DistributionMap(), 1, 0,
                    MFInfo());
                level_masks[idim].setVal(1);
            }
        }

        set_adapt_inflow_masks(
            lev, vels_vec, masks, level_masks, bc_types, domain,
            include_bndry_corners);

        const Real* a_dx = geom[lev].CellSize();
        Real inf_lev = 0.0, out_lev = 0.0, pa_lev = 0.0;
        compute_adapt_inflow_fluxes(
            lev, vels_vec, masks, a_dx, inf_lev, out_lev, pa_lev,
            include_bndry_corners);
        influx       += inf_lev;
        outflux      += out_lev;
        passive_area += pa_lev;
    }

    // nothing to do if outflux does not exceed influx
    if (outflux <= influx + small_vel) { return; }

    if (passive_area < small_vel) {
        amrex::Abort(
            "enforceAdaptInflowSolvability: outflux exceeds influx but no "
            "passive cells are available to supply the deficit inflow");
    }

    const Real v_passive = (outflux - influx) / passive_area;

    for (int lev = 0; lev < nlevs; ++lev) {
        apply_passive_inflow(
            lev, vels_vec, bc_types, geom[lev].Domain(), v_passive,
            include_bndry_corners);
    }
}

} // namespace kynema_sgf

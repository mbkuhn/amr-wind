#include "src/projection/AdaptInflowSolvability.H"

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
constexpr int AI_INFLOW_TAG      = -1;
constexpr int AI_OUTFLOW_TAG     = +1;
constexpr int AI_PASSIVE_TAG     = +2;
constexpr int AI_EXTRAP_OUT_TAG  = +3;

constexpr Real small_vel = 1.e-8_rt;

/** Tag each adapt_inflow boundary cell as inflow, outflow, extrap_out, or
 *  passive.
 *
 *  For a low boundary, the categorisation is:
 *    inflow:      vel_boundary > 0  (points into domain)
 *    outflow:     vel_boundary < 0  (boundary value itself points outward)
 *    extrap_out:  vel_boundary == 0 AND vel_interior < 0 (outward, but the
 *                 boundary value has not been extrapolated yet)
 *    passive:     vel_boundary == 0 AND vel_interior >= 0
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
    const bool corners,
    const iMultiFab* terrain_blank_mf)
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
            const auto terrain_arr =
                terrain_blank_mf ? terrain_blank_mf->const_array(mfi)
                                 : Array4<int const>();

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const IntVect iv{AMREX_D_DECL(i, j, k)};
                const IntVect lm_idx =
                    amrex::max(amrex::min(iv, ungrown_bg), ungrown_sm);
                if (level_mask_arr(lm_idx) != 1) { return; }

                if (terrain_arr) {
                    IntVect terrain_iv{AMREX_D_DECL(i, j, k)};
                    terrain_iv[dir] =
                        oriIsLow ? domain.smallEnd(dir) : domain.bigEnd(dir);
                    if (terrain_arr(terrain_iv) == 1) { return; }
                }

                const Real vb = vel_arr(i, j, k);
                const Real vi = oriIsLow ? vel_arr(i + di, j + dj, k + dk)
                                         : vel_arr(i - di, j - dj, k - dk);

                if ((oriIsLow && vb > 0) || (oriIsHigh && vb < 0)) {
                    mask_arr(i, j, k) = AI_INFLOW_TAG;
                } else if ((oriIsLow && vb < 0) || (oriIsHigh && vb > 0)) {
                    mask_arr(i, j, k) = AI_OUTFLOW_TAG;
                } else if ((oriIsLow && vi < 0) || (oriIsHigh && vi > 0)) {
                    mask_arr(i, j, k) = AI_EXTRAP_OUT_TAG;
                } else {
                    mask_arr(i, j, k) = AI_PASSIVE_TAG;
                }
            });
        }
    }
}

/** Accumulate influx, the outflow and extrap_out fluxes, the directional
 *  (signed) outflux vector, and the passive face area of each individual
 *  boundary (orientation) from the masks.
 *
 *  outflow_flux sums abs(vel_boundary) over outflow-tagged cells, while
 *  extrap_out_flux sums abs(vel_interior) over extrap_out-tagged cells since
 *  their boundary value is still zero. outflux_vector is the signed
 *  combination of both, giving the net direction the outflow is moving in.
 *
 *  passive_area is indexed the same way as bc_types, i.e. orientation index
 *  `dim` is the low side of direction `dim` and `dim + AMREX_SPACEDIM` is the
 *  high side.
 */
void compute_adapt_inflow_fluxes(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Box& domain,
    const Real* a_dx,
    Real& influx,
    Real& outflow_flux,
    Real& extrap_out_flux,
    GpuArray<Real, AMREX_SPACEDIM>& outflux_vector,
    GpuArray<Real, AMREX_SPACEDIM * 2>& passive_area,
    const bool corners)
{
    influx = 0.0; outflow_flux = 0.0; extrap_out_flux = 0.0;
    for (int i = 0; i < AMREX_SPACEDIM; i++) { outflux_vector[i] = 0.0; }
    for (int i = 0; i < AMREX_SPACEDIM * 2; i++) { passive_area[i] = 0.0; }

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        const Real ds = a_dx[(idim + 1) % AMREX_SPACEDIM]
                      * a_dx[(idim + 2) % AMREX_SPACEDIM];

        const auto& vel_mf = vels_vec[lev][idim];

        const IndexType::CellIndex dir_idx_type =
            (vel_mf->ixType()).ixType(idim);
        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(idim) - 1
                            : domain.smallEnd(idim);
        const int dhi = domain.bigEnd(idim) + 1;

        // unit step from boundary toward domain interior
        const int di = (idim == 0) ? 1 : 0;
        const int dj = (idim == 1) ? 1 : 0;
        const int dk = (idim == 2) ? 1 : 0;

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

        outflow_flux += ds *
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

        extrap_out_flux += ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                if (mask_ma[box_no](i, j, k) != AI_EXTRAP_OUT_TAG) {
                    return {0.0_rt};
                }
                const int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
                const Real vi = (idx == dlo)
                                    ? vel_ma[box_no](i + di, j + dj, k + dk)
                                    : vel_ma[box_no](i - di, j - dj, k - dk);
                return {std::abs(vi)};
            });

        // signed sum gives the net direction the outflow is moving in
        outflux_vector[idim] = ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                const int tag = mask_ma[box_no](i, j, k);
                if (tag == AI_OUTFLOW_TAG) {
                    return {vel_ma[box_no](i, j, k)};
                }
                if (tag == AI_EXTRAP_OUT_TAG) {
                    const int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
                    const Real vi =
                        (idx == dlo)
                            ? vel_ma[box_no](i + di, j + dj, k + dk)
                            : vel_ma[box_no](i - di, j - dj, k - dk);
                    return {vi};
                }
                return {0.0_rt};
            });

        passive_area[idim] = ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                if (mask_ma[box_no](i, j, k) != AI_PASSIVE_TAG) {
                    return {0.0_rt};
                }
                const int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
                return {(idx == dlo) ? 1.0_rt : 0.0_rt};
            });

        passive_area[idim + AMREX_SPACEDIM] = ds *
            ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real>
            {
                if (mask_ma[box_no](i, j, k) != AI_PASSIVE_TAG) {
                    return {0.0_rt};
                }
                const int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
                return {(idx == dhi) ? 1.0_rt : 0.0_rt};
            });
    }

    ParallelDescriptor::ReduceRealSum(influx);
    ParallelDescriptor::ReduceRealSum(outflow_flux);
    ParallelDescriptor::ReduceRealSum(extrap_out_flux);
    ParallelDescriptor::ReduceRealSum(outflux_vector.data(), AMREX_SPACEDIM);
    ParallelDescriptor::ReduceRealSum(
        passive_area.data(), AMREX_SPACEDIM * 2);
}

/** Set boundary velocity for passive cells to achieve mass balance.
 *
 *  Only boundaries marked active in \p selected_lo / \p selected_hi (chosen
 *  by the caller based on the direction of the net outflux vector) are
 *  modified; passive cells on other adapt_inflow boundaries are left as-is.
 *  Cells whose adjacent interior cell is terrain-blanked are skipped.
 *
 *  Passive cells are identified by the same two-velocity check used in
 *  set_adapt_inflow_masks rather than reading the mask iMultiFab, so that
 *  this function can be called in a second pass after all fluxes are known.
 */
void apply_passive_flux(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const Real v_corr,
    const GpuArray<bool, AMREX_SPACEDIM>& selected_lo,
    const GpuArray<bool, AMREX_SPACEDIM>& selected_hi,
    const bool corners,
    const iMultiFab* terrain_blank_mf)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) { continue; }

        const int dir = ori.coordDir();
        const bool oriIsLow  = ori.isLow();
        const bool oriIsHigh = ori.isHigh();

        if ((oriIsLow && !selected_lo[dir]) ||
            (oriIsHigh && !selected_hi[dir])) {
            continue;
        }

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
        const Real v_inward = oriIsLow ? v_corr : -v_corr;

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
            const auto terrain_arr =
                terrain_blank_mf ? terrain_blank_mf->const_array(mfi)
                                 : Array4<int const>();

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (terrain_arr) {
                    IntVect terrain_iv{AMREX_D_DECL(i, j, k)};
                    terrain_iv[dir] =
                        oriIsLow ? domain.smallEnd(dir) : domain.bigEnd(dir);
                    if (terrain_arr(terrain_iv) == 1) { return; }
                }

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

/** Extrapolate extrap_out cells from the interior, optionally scaled.
 *
 *  Each extrap_out cell is re-identified locally (boundary velocity is zero
 *  and the adjacent interior velocity points outward) and its boundary
 *  velocity is set to vel_interior * alpha. This is always called with at
 *  least alpha = 1 so that extrap_out boundary values are always populated;
 *  alpha is only different from 1 when additional outflow is needed to
 *  match influx. Cells whose adjacent interior cell is terrain-blanked are
 *  left untouched.
 */
void apply_extrap_out_scale(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const Real alpha,
    const bool corners,
    const iMultiFab* terrain_blank_mf)
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
            const auto terrain_arr =
                terrain_blank_mf ? terrain_blank_mf->const_array(mfi)
                                 : Array4<int const>();

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (terrain_arr) {
                    IntVect terrain_iv{AMREX_D_DECL(i, j, k)};
                    terrain_iv[dir] =
                        oriIsLow ? domain.smallEnd(dir) : domain.bigEnd(dir);
                    if (terrain_arr(terrain_iv) == 1) { return; }
                }

                const Real vb = vel_arr(i, j, k);
                const Real vi = oriIsLow ? vel_arr(i + di, j + dj, k + dk)
                                         : vel_arr(i - di, j - dj, k - dk);

                const bool is_extrap_out =
                    (vb == 0) &&
                    ((oriIsLow && vi < 0) || (oriIsHigh && vi > 0));

                if (is_extrap_out) { vel_arr(i, j, k) = vi * alpha; }
            });
        }
    }
}

} // file-local namespace

void enforceAdaptInflowSolvability(
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Vector<Geometry>& geom,
    const bool include_bndry_corners,
    const IntField* terrain_blank)
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
    Real influx = 0.0, outflow_flux = 0.0, extrap_out_flux = 0.0;
    GpuArray<Real, AMREX_SPACEDIM> outflux_vector{};
    GpuArray<Real, AMREX_SPACEDIM * 2> passive_area{};
    for (int i = 0; i < AMREX_SPACEDIM; i++) { outflux_vector[i] = 0.0; }
    for (int i = 0; i < AMREX_SPACEDIM * 2; i++) { passive_area[i] = 0.0; }

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
            include_bndry_corners,
            terrain_blank ? &(*terrain_blank)(lev) : nullptr);

        const Real* a_dx = geom[lev].CellSize();
        Real inf_lev = 0.0, outf_lev = 0.0, extrap_lev = 0.0;
        GpuArray<Real, AMREX_SPACEDIM> outvec_lev{};
        GpuArray<Real, AMREX_SPACEDIM * 2> pa_lev{};
        compute_adapt_inflow_fluxes(
            lev, vels_vec, masks, domain, a_dx, inf_lev, outf_lev,
            extrap_lev, outvec_lev, pa_lev, include_bndry_corners);
        influx          += inf_lev;
        outflow_flux    += outf_lev;
        extrap_out_flux += extrap_lev;
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            outflux_vector[i] += outvec_lev[i];
        }
        for (int i = 0; i < AMREX_SPACEDIM * 2; i++) {
            passive_area[i] += pa_lev[i];
        }
    }

    // net imbalance to fix: positive means influx must be added, negative
    // means outflux must be added
    const Real total_outflux = outflow_flux + extrap_out_flux;
    const Real deficit = total_outflux - influx;

    // extrap_out cells always get their boundary value extrapolated from the
    // adjacent interior velocity (alpha = 1); they are additionally scaled
    // only when more outflow is needed to match influx (deficit < 0).
    Real alpha = 1.0_rt;
    if (deficit < -small_vel) {
        if (extrap_out_flux < small_vel) {
            amrex::Abort(
                "enforceAdaptInflowSolvability: no extrap_out cells are "
                "available on any adapt_inflow boundary to supply the "
                "required additional outflow");
        }
        alpha = (influx - outflow_flux) / extrap_out_flux;
    }

    for (int lev = 0; lev < nlevs; ++lev) {
        apply_extrap_out_scale(
            lev, vels_vec, bc_types, geom[lev].Domain(), alpha,
            include_bndry_corners,
            terrain_blank ? &(*terrain_blank)(lev) : nullptr);
    }

    // Balance already satisfied (or resolved via the extrap_out scaling
    // above); no need to touch passive cells.
    if (deficit <= small_vel) { return; }

    // Additional inflow is needed: add a uniform inward velocity to passive
    // cells upstream of the net outflux vector.
    GpuArray<bool, AMREX_SPACEDIM> selected_lo{};
    GpuArray<bool, AMREX_SPACEDIM> selected_hi{};
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        selected_lo[idim] = outflux_vector[idim] > 0;
        selected_hi[idim] = outflux_vector[idim] < 0;
    }

    Real selected_area = 0.0;
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        if (selected_lo[idim]) { selected_area += passive_area[idim]; }
        if (selected_hi[idim]) {
            selected_area += passive_area[idim + AMREX_SPACEDIM];
        }
    }

    // Failsafe: no passive area on the correct upstream/downstream
    // boundaries, so fall back to spreading the correction over every
    // adapt_inflow passive boundary instead of aborting.
    if (selected_area < small_vel) {
        amrex::Print() << "enforceAdaptInflowSolvability: no passive area "
                          "upstream/downstream of the net outflux direction; "
                          "falling back to all passive boundaries\n";
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            selected_lo[idim] = true;
            selected_hi[idim] = true;
        }
        selected_area = 0.0;
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            selected_area +=
                passive_area[idim] + passive_area[idim + AMREX_SPACEDIM];
        }
    }

    if (selected_area < small_vel) {
        amrex::Abort(
            "enforceAdaptInflowSolvability: no passive cells are available "
            "on any adapt_inflow boundary to balance the flux");
    }

    const Real v_corr = deficit / selected_area;

    for (int lev = 0; lev < nlevs; ++lev) {
        apply_passive_flux(
            lev, vels_vec, bc_types, geom[lev].Domain(), v_corr, selected_lo,
            selected_hi, include_bndry_corners,
            terrain_blank ? &(*terrain_blank)(lev) : nullptr);
    }
}

} // namespace kynema_sgf

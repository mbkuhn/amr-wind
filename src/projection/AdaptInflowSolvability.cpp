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

// Mask values used to tag each boundary cell category. 0 (the iMultiFab
// default) means "not yet classified".
constexpr int AI_INFLOW_TAG = -1;
constexpr int AI_OUTFLOW_TAG = +1;
constexpr int AI_PASSIVE_TAG = +2;
constexpr int AI_EXTRAP_OUT_TAG = +3;

constexpr Real small_vel = 1.e-8_rt;

// Minimum vof for a cell to be considered wet when classifying boundary cells
constexpr Real vof_threshold = 1.e-12_rt;

/** Classify an adapt_inflow boundary cell as inflow or outflow.
 *
 *  For a low boundary:
 *    inflow:   vel_boundary > 0 (points into domain) AND the boundary cell
 *              is wet (vof_boundary > vof_threshold, if supplied)
 *    outflow:  vel_boundary < 0 (boundary value itself points outward) AND
 *              the adjacent interior cell is wet
 *              (vof_interior > vof_threshold, if supplied)
 *
 *  The sign conventions are reversed for high boundaries. Cells matching
 *  neither condition return 0 (unclassified); these are resolved into
 *  extrap_out or passive in a second pass once the net outflow direction is
 *  known.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int classify_inflow_outflow(
    const Real vel_boundary,
    const bool ori_is_low,
    const bool ori_is_high,
    const bool vof_boundary_wet,
    const bool vof_interior_wet)
{
    if (((ori_is_low && vel_boundary > 0) ||
         (ori_is_high && vel_boundary < 0)) &&
        vof_boundary_wet) {
        return AI_INFLOW_TAG;
    }
    if (((ori_is_low && vel_boundary < 0) ||
         (ori_is_high && vel_boundary > 0)) &&
        vof_interior_wet) {
        return AI_OUTFLOW_TAG;
    }
    return 0;
}

/** First classification pass: tag inflow and outflow cells only, leaving
 *  everything else (including terrain-blanked or finer-level-covered cells)
 *  at the default mask value of 0.
 */
void set_inflow_outflow_masks(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Array<iMultiFab, AMREX_SPACEDIM>& level_masks,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const bool corners,
    const iMultiFab* terrain_blank_mf,
    const MultiFab* vof_mf)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) {
            continue;
        }

        const int dir = ori.coordDir();
        const bool oriIsLow = ori.isLow();
        const bool oriIsHigh = ori.isHigh();

        const auto& vel_mf = vels_vec[lev][dir];
        auto& mask = masks[dir];
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
                if (box.smallEnd(t1) == domain.smallEnd(t1)) {
                    box.growLo(t1, 1);
                }
                if (box.bigEnd(t1) == domain.bigEnd(t1)) {
                    box.growHi(t1, 1);
                }
#if (AMREX_SPACEDIM == 3)
                const int t2 = (dir + 2) % AMREX_SPACEDIM;
                if (box.smallEnd(t2) == domain.smallEnd(t2)) {
                    box.growLo(t2, 1);
                }
                if (box.bigEnd(t2) == domain.bigEnd(t2)) {
                    box.growHi(t2, 1);
                }
#endif
            }

            const bool at_lo_bndry = oriIsLow && (box.smallEnd(dir) == dlo);
            const bool at_hi_bndry = oriIsHigh && (box.bigEnd(dir) == dhi);
            if (!at_lo_bndry && !at_hi_bndry) {
                continue;
            }

            Box box2d(box);
            box2d.setRange(dir, bndry);

            const auto vel_arr = vel_mf->array(mfi);
            auto mask_arr = mask.array(mfi);
            const auto level_mask_arr = level_mask.const_array(mfi);
            const auto terrain_arr = terrain_blank_mf
                                         ? terrain_blank_mf->const_array(mfi)
                                         : Array4<int const>();
            const auto vof_arr =
                vof_mf ? vof_mf->const_array(mfi) : Array4<Real const>();

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const IntVect iv{AMREX_D_DECL(i, j, k)};
                const IntVect lm_idx =
                    amrex::max(amrex::min(iv, ungrown_bg), ungrown_sm);
                if (level_mask_arr(lm_idx) != 1) {
                    return;
                }

                if (terrain_arr) {
                    IntVect terrain_iv{AMREX_D_DECL(i, j, k)};
                    terrain_iv[dir] =
                        oriIsLow ? domain.smallEnd(dir) : domain.bigEnd(dir);
                    if (terrain_arr(terrain_iv) == 1) {
                        return;
                    }
                }

                const Real vb = vel_arr(i, j, k);

                const bool vof_bndry_wet =
                    !vof_arr || (vof_arr(i, j, k) > vof_threshold);
                const bool vof_interior_wet =
                    !vof_arr || ((oriIsLow ? vof_arr(i + di, j + dj, k + dk)
                                           : vof_arr(i - di, j - dj, k - dk)) >
                                 vof_threshold);

                mask_arr(i, j, k) = classify_inflow_outflow(
                    vb, oriIsLow, oriIsHigh, vof_bndry_wet, vof_interior_wet);
            });
        }
    }
}

/** Sum the signed boundary velocity over outflow-tagged cells only, giving
 *  the net direction the (already-identified) outflow is moving in.
 */
void compute_outflow_vector(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Real* a_dx,
    GpuArray<Real, AMREX_SPACEDIM>& outflow_vector,
    const bool corners)
{
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
        outflow_vector[i] = 0.0;
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        const Real ds = a_dx[(idim + 1) % AMREX_SPACEDIM] *
                        a_dx[(idim + 2) % AMREX_SPACEDIM];
        const auto& vel_mf = vels_vec[lev][idim];

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
        const auto vel_ma = vel_mf->const_arrays();
        const auto mask_ma = mask.const_arrays();

        outflow_vector[idim] =
            ds * ParReduce(
                     TypeList<ReduceOpSum>{}, TypeList<Real>{}, *vel_mf, ngrow,
                     [=] AMREX_GPU_DEVICE(
                         int box_no, int i, int j,
                         int k) noexcept -> GpuTuple<Real> {
                         if (mask_ma[box_no](i, j, k) == AI_OUTFLOW_TAG) {
                             return {vel_ma[box_no](i, j, k)};
                         }
                         return {0.0_rt};
                     });
    }

    ParallelDescriptor::ReduceRealSum(outflow_vector.data(), AMREX_SPACEDIM);
}

/** Second classification pass: resolve any still-unclassified boundary cell
 *  (mask value 0) into extrap_out or passive.
 *
 *  A boundary is either entirely eligible for extrap_out, if it is
 *  downstream of the net outflow direction (per \p allow_outflow_lo /
 *  \p allow_outflow_hi), or entirely eligible for passive otherwise; no
 *  further per-cell velocity check is performed here.
 */
void set_extrap_out_or_passive_masks(
    Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Array<iMultiFab, AMREX_SPACEDIM>& level_masks,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const bool corners,
    const iMultiFab* terrain_blank_mf,
    const GpuArray<bool, AMREX_SPACEDIM>& allow_outflow_lo,
    const GpuArray<bool, AMREX_SPACEDIM>& allow_outflow_hi)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) {
            continue;
        }

        const int dir = ori.coordDir();
        const bool oriIsLow = ori.isLow();
        const bool oriIsHigh = ori.isHigh();
        const int new_tag =
            (oriIsLow ? allow_outflow_lo[dir] : allow_outflow_hi[dir])
                ? AI_EXTRAP_OUT_TAG
                : AI_PASSIVE_TAG;

        auto& mask = masks[dir];
        const auto& level_mask = level_masks[dir];

        const IndexType::CellIndex dir_idx_type = mask.ixType().ixType(dir);

        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(dir) - 1
                            : domain.smallEnd(dir);
        const int dhi = domain.bigEnd(dir) + 1;
        const int bndry = oriIsLow ? dlo : dhi;

        for (MFIter mfi(mask, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box box = mfi.tilebox();
            const IntVect ungrown_sm = box.smallEnd();
            const IntVect ungrown_bg = box.bigEnd();

            if (dir_idx_type == IndexType::CellIndex::CELL) {
                box.grow(dir, 1);
            }
            if (corners) {
                const int t1 = (dir + 1) % AMREX_SPACEDIM;
                if (box.smallEnd(t1) == domain.smallEnd(t1)) {
                    box.growLo(t1, 1);
                }
                if (box.bigEnd(t1) == domain.bigEnd(t1)) {
                    box.growHi(t1, 1);
                }
#if (AMREX_SPACEDIM == 3)
                const int t2 = (dir + 2) % AMREX_SPACEDIM;
                if (box.smallEnd(t2) == domain.smallEnd(t2)) {
                    box.growLo(t2, 1);
                }
                if (box.bigEnd(t2) == domain.bigEnd(t2)) {
                    box.growHi(t2, 1);
                }
#endif
            }

            const bool at_lo_bndry = oriIsLow && (box.smallEnd(dir) == dlo);
            const bool at_hi_bndry = oriIsHigh && (box.bigEnd(dir) == dhi);
            if (!at_lo_bndry && !at_hi_bndry) {
                continue;
            }

            Box box2d(box);
            box2d.setRange(dir, bndry);

            auto mask_arr = mask.array(mfi);
            const auto level_mask_arr = level_mask.const_array(mfi);
            const auto terrain_arr = terrain_blank_mf
                                         ? terrain_blank_mf->const_array(mfi)
                                         : Array4<int const>();

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const IntVect iv{AMREX_D_DECL(i, j, k)};
                const IntVect lm_idx =
                    amrex::max(amrex::min(iv, ungrown_bg), ungrown_sm);
                if (level_mask_arr(lm_idx) != 1) {
                    return;
                }

                if (terrain_arr) {
                    IntVect terrain_iv{AMREX_D_DECL(i, j, k)};
                    terrain_iv[dir] =
                        oriIsLow ? domain.smallEnd(dir) : domain.bigEnd(dir);
                    if (terrain_arr(terrain_iv) == 1) {
                        return;
                    }
                }

                if (mask_arr(i, j, k) != 0) {
                    return;
                }

                mask_arr(i, j, k) = new_tag;
            });
        }
    }
}

/** Accumulate influx, the outflow and extrap_out fluxes, and the total
 *  passive face area from the (fully classified) masks.
 *
 *  outflow_flux sums abs(vel_boundary) over outflow-tagged cells, while
 *  extrap_out_flux sums abs(vel_interior) over extrap_out-tagged cells since
 *  their boundary value has not been extrapolated yet.
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
    Real& passive_area,
    const bool corners)
{
    influx = 0.0;
    outflow_flux = 0.0;
    extrap_out_flux = 0.0;
    passive_area = 0.0;

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        const Real ds = a_dx[(idim + 1) % AMREX_SPACEDIM] *
                        a_dx[(idim + 2) % AMREX_SPACEDIM];

        const auto& vel_mf = vels_vec[lev][idim];

        const IndexType::CellIndex dir_idx_type =
            (vel_mf->ixType()).ixType(idim);
        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(idim) - 1
                            : domain.smallEnd(idim);

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
        const auto vel_ma = vel_mf->const_arrays();
        const auto mask_ma = mask.const_arrays();

        influx +=
            ds * ParReduce(
                     TypeList<ReduceOpSum>{}, TypeList<Real>{}, *vel_mf, ngrow,
                     [=] AMREX_GPU_DEVICE(
                         int box_no, int i, int j,
                         int k) noexcept -> GpuTuple<Real> {
                         if (mask_ma[box_no](i, j, k) == AI_INFLOW_TAG) {
                             return {std::abs(vel_ma[box_no](i, j, k))};
                         }
                         return {0.0_rt};
                     });

        outflow_flux +=
            ds * ParReduce(
                     TypeList<ReduceOpSum>{}, TypeList<Real>{}, *vel_mf, ngrow,
                     [=] AMREX_GPU_DEVICE(
                         int box_no, int i, int j,
                         int k) noexcept -> GpuTuple<Real> {
                         if (mask_ma[box_no](i, j, k) == AI_OUTFLOW_TAG) {
                             return {std::abs(vel_ma[box_no](i, j, k))};
                         }
                         return {0.0_rt};
                     });

        extrap_out_flux +=
            ds *
            ParReduce(
                TypeList<ReduceOpSum>{}, TypeList<Real>{}, *vel_mf, ngrow,
                [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
                    -> GpuTuple<Real> {
                    if (mask_ma[box_no](i, j, k) != AI_EXTRAP_OUT_TAG) {
                        return {0.0_rt};
                    }
                    const int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
                    Real vi = (idx == dlo)
                                  ? vel_ma[box_no](i + di, j + dj, k + dk)
                                  : vel_ma[box_no](i - di, j - dj, k - dk);
                    // Only count outflow
                    vi = (idx == dlo) ? amrex::min<Real>(vi, 0.0_rt)
                                      : amrex::max<Real>(vi, 0.0_rt);
                    return {std::abs(vi)};
                });

        passive_area +=
            ds *
            ParReduce(
                TypeList<ReduceOpSum>{}, TypeList<Real>{}, *vel_mf, ngrow,
                [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
                    -> GpuTuple<Real> {
                    return {
                        (mask_ma[box_no](i, j, k) == AI_PASSIVE_TAG) ? 1.0_rt
                                                                     : 0.0_rt};
                });
    }

    ParallelDescriptor::ReduceRealSum(influx);
    ParallelDescriptor::ReduceRealSum(outflow_flux);
    ParallelDescriptor::ReduceRealSum(extrap_out_flux);
    ParallelDescriptor::ReduceRealSum(passive_area);
}

/** Set boundary velocity for passive cells to achieve mass balance.
 *
 *  Passive cells are read directly from the persisted category mask
 *  computed by set_extrap_out_or_passive_masks.
 */
void apply_passive_flux(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Real v_corr,
    const bool corners)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) {
            continue;
        }

        const int dir = ori.coordDir();
        const bool oriIsLow = ori.isLow();
        const bool oriIsHigh = ori.isHigh();

        const auto& vel_mf = vels_vec[lev][dir];
        const auto& mask = masks[dir];

        const IndexType::CellIndex dir_idx_type =
            (vel_mf->ixType()).ixType(dir);

        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(dir) - 1
                            : domain.smallEnd(dir);
        const int dhi = domain.bigEnd(dir) + 1;
        const int bndry = oriIsLow ? dlo : dhi;

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

            if ((oriIsLow && (box.smallEnd(dir) != dlo)) ||
                (oriIsHigh && (box.bigEnd(dir) != dhi))) {
                continue;
            }

            Box box2d(box);
            box2d.setRange(dir, bndry);
            auto vel_arr = vel_mf->array(mfi);
            const auto mask_arr = mask.const_array(mfi);

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (mask_arr(i, j, k) == AI_PASSIVE_TAG) {
                    vel_arr(i, j, k) = v_inward;
                }
            });
        }
    }
}

/** Extrapolate extrap_out cells from the interior, optionally scaled.
 *
 *  extrap_out cells are read directly from the persisted category mask; the
 *  boundary velocity is set to vel_interior * alpha. This is always called
 *  with at least alpha = 1 so that extrap_out boundary values are always
 *  populated; alpha differs from 1 only when additional outflow is needed to
 *  match influx.
 */
void apply_extrap_out_scale(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Box& domain,
    const Array<iMultiFab, AMREX_SPACEDIM>& masks,
    const Real alpha,
    const bool corners)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        if (bc_types[ori] != BC::adapt_inflow) {
            continue;
        }

        const int dir = ori.coordDir();
        const bool oriIsLow = ori.isLow();
        const bool oriIsHigh = ori.isHigh();

        const auto& vel_mf = vels_vec[lev][dir];
        const auto& mask = masks[dir];

        const IndexType::CellIndex dir_idx_type =
            (vel_mf->ixType()).ixType(dir);

        const int dlo = (dir_idx_type == IndexType::CellIndex::CELL)
                            ? domain.smallEnd(dir) - 1
                            : domain.smallEnd(dir);
        const int dhi = domain.bigEnd(dir) + 1;
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

            if ((oriIsLow && (box.smallEnd(dir) != dlo)) ||
                (oriIsHigh && (box.bigEnd(dir) != dhi))) {
                continue;
            }

            Box box2d(box);
            box2d.setRange(dir, bndry);
            auto vel_arr = vel_mf->array(mfi);
            const auto mask_arr = mask.const_array(mfi);

            ParallelFor(box2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (mask_arr(i, j, k) != AI_EXTRAP_OUT_TAG) {
                    return;
                }

                Real vi = oriIsLow ? vel_arr(i + di, j + dj, k + dk)
                                   : vel_arr(i - di, j - dj, k - dk);
                // Only retain outflow
                vi = oriIsLow ? amrex::min<Real>(vi, 0.0_rt)
                              : amrex::max<Real>(vi, 0.0_rt);
                vel_arr(i, j, k) = vi * alpha;
            });
        }
    }
}

} // namespace

void enforceAdaptInflowSolvability(
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const Vector<Geometry>& geom,
    const bool include_bndry_corners,
    const IntField* terrain_blank,
    const Field* vof)
{
    bool has_adapt_inflow = false;
    for (OrientationIter oit; oit != nullptr; ++oit) {
        if (bc_types[oit()] == BC::adapt_inflow) {
            has_adapt_inflow = true;
            break;
        }
    }
    if (!has_adapt_inflow) {
        return;
    }

    const int nlevs = static_cast<int>(vels_vec.size());

    // Category masks and finer-level coverage masks, persisted per level
    // across both classification passes and the later flux corrections.
    Vector<Array<iMultiFab, AMREX_SPACEDIM>> masks_vec(nlevs);
    Vector<Array<iMultiFab, AMREX_SPACEDIM>> level_masks_vec(nlevs);

    for (int lev = 0; lev < nlevs; ++lev) {
        auto& masks = masks_vec[lev];
        auto& level_masks = level_masks_vec[lev];

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

        set_inflow_outflow_masks(
            lev, vels_vec, masks, level_masks, bc_types, geom[lev].Domain(),
            include_bndry_corners,
            terrain_blank ? &(*terrain_blank)(lev) : nullptr,
            vof ? &(*vof)(lev) : nullptr);
    }

    // Net direction of the (already-identified) outflow.
    GpuArray<Real, AMREX_SPACEDIM> outflow_vector{};
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
        outflow_vector[i] = 0.0;
    }
    for (int lev = 0; lev < nlevs; ++lev) {
        GpuArray<Real, AMREX_SPACEDIM> ov_lev{};
        compute_outflow_vector(
            lev, vels_vec, masks_vec[lev], geom[lev].CellSize(), ov_lev,
            include_bndry_corners);
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            outflow_vector[i] += ov_lev[i];
        }
    }

    // A boundary allows outflow if it is downstream of the net outflow
    // direction (e.g. xhi if the vector has a positive x component); its
    // remaining cells become extrap_out. The opposite boundary's remaining
    // cells become passive.
    GpuArray<bool, AMREX_SPACEDIM> allow_outflow_lo{};
    GpuArray<bool, AMREX_SPACEDIM> allow_outflow_hi{};
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        allow_outflow_lo[idim] = outflow_vector[idim] < 0;
        allow_outflow_hi[idim] = outflow_vector[idim] > 0;
    }

    Real influx = 0.0, outflow_flux = 0.0, extrap_out_flux = 0.0;
    Real passive_area = 0.0;

    for (int lev = 0; lev < nlevs; ++lev) {
        set_extrap_out_or_passive_masks(
            masks_vec[lev], level_masks_vec[lev], bc_types, geom[lev].Domain(),
            include_bndry_corners,
            terrain_blank ? &(*terrain_blank)(lev) : nullptr, allow_outflow_lo,
            allow_outflow_hi);

        Real inf_lev = 0.0, outf_lev = 0.0, extrap_lev = 0.0, pa_lev = 0.0;
        compute_adapt_inflow_fluxes(
            lev, vels_vec, masks_vec[lev], geom[lev].Domain(),
            geom[lev].CellSize(), inf_lev, outf_lev, extrap_lev, pa_lev,
            include_bndry_corners);
        influx += inf_lev;
        outflow_flux += outf_lev;
        extrap_out_flux += extrap_lev;
        passive_area += pa_lev;
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
            lev, vels_vec, bc_types, geom[lev].Domain(), masks_vec[lev], alpha,
            include_bndry_corners);
    }

    // Even if balance is already satisfied, make sure to disable outflow at
    // passive cells

    // Additional inflow is needed. Passive cells only exist on boundaries
    // opposite the net outflow direction (by construction of the second
    // classification pass), so no further directional selection is needed.
    if (deficit > small_vel && passive_area < small_vel) {
        amrex::Abort(
            "enforceAdaptInflowSolvability: no passive cells are available "
            "on any adapt_inflow boundary to balance the flux");
    }

    const Real v_corr = passive_area > small_vel
                            ? amrex::max<Real>(deficit, 0.0_rt) / passive_area
                            : 0.0_rt;

    for (int lev = 0; lev < nlevs; ++lev) {
        apply_passive_flux(
            lev, vels_vec, bc_types, geom[lev].Domain(), masks_vec[lev], v_corr,
            include_bndry_corners);
    }
}

} // namespace kynema_sgf

#ifndef ADAPT_INFLOW_SOLVABILITY_H
#define ADAPT_INFLOW_SOLVABILITY_H

#include "src/incflo_enums.H"
#include "src/core/Field.H"
#include "src/core/IntField.H"
#include "AMReX_Array.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"

namespace kynema_sgf {

/** Enforce mass conservation for the adapt_inflow boundary condition.
 *
 *  Classifies each adapt_inflow boundary cell into one of four categories:
 *    - inflow:      velocity at the boundary points into the domain, and
 *                   (if a vof field is supplied) the boundary cell is wet
 *    - outflow:     velocity at the boundary itself points outward, and
 *                   (if a vof field is supplied) the adjacent interior cell
 *                   is wet
 *    - extrap_out:  velocity just inside the domain points outward (no
 *                   check on the boundary velocity or on vof)
 *    - passive:     the remaining cells (neither inflow, outflow, nor
 *                   extrap_out)
 *
 *  The total outflux is the sum of the outflow and extrap_out contributions.
 *  extrap_out boundary values are always extrapolated from the adjacent
 *  interior velocity, regardless of the flux balance. If influx exceeds the
 *  total outflux, that extrapolated value is additionally scaled by a single
 *  factor so that the total outflux matches influx. If the total outflux
 *  exceeds influx instead, a uniform inward velocity is applied to passive
 *  cells on boundaries upstream of the net (signed) outflux vector (e.g. xlo
 *  if the vector has a positive x component).
 *
 *  \param vels_vec     Per-level, per-direction normal velocity MultiFabs
 *  \param bc_types     Incflo BC type for each orientation
 *  \param geom         AMReX geometry vector (one per level)
 *  \param include_bndry_corners  If true, include corner cells in flux sums
 *  \param terrain_blank  Optional terrain blanking field. Boundary locations
 *                        whose adjacent interior cell is terrain-blanked
 *                        (value 1) are excluded from area calculations and
 *                        velocity modifications.
 *  \param vof  Optional volume-of-fluid field. When supplied, inflow and
 *              outflow tags additionally require vof > 1e-12 (at the
 *              boundary cell for inflow, at the adjacent interior cell for
 *              outflow).
 */
void enforceAdaptInflowSolvability(
    const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
        vels_vec,
    const amrex::GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const amrex::Vector<amrex::Geometry>& geom,
    bool include_bndry_corners = false,
    const IntField* terrain_blank = nullptr,
    const Field* vof = nullptr);

} // namespace kynema_sgf

#endif /* ADAPT_INFLOW_SOLVABILITY_H */

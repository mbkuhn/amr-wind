#ifndef ADAPT_INFLOW_SOLVABILITY_H
#define ADAPT_INFLOW_SOLVABILITY_H

#include "src/incflo_enums.H"
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
 *    - inflow:      velocity at the boundary points into the domain
 *    - outflow:     velocity at the boundary itself points outward
 *    - extrap_out:  boundary velocity is zero, but velocity just inside the
 *                   domain points outward (i.e. an outflow cell whose
 *                   boundary value has not yet been extrapolated)
 *    - passive:     the remaining cells (neither inflow, outflow, nor
 *                   extrap_out)
 *
 *  The total outflux is the sum of the outflow and extrap_out contributions.
 *  If the total outflux exceeds influx, a uniform inward velocity is applied
 *  only to passive cells on boundaries upstream of the net (signed) outflux
 *  vector (e.g. xlo if the vector has a positive x component). If influx
 *  exceeds the total outflux, the extrap_out cells are instead extrapolated
 *  from the adjacent interior velocity and scaled by a single factor so that
 *  the total outflux matches influx.
 *
 *  \param vels_vec     Per-level, per-direction normal velocity MultiFabs
 *  \param bc_types     Incflo BC type for each orientation
 *  \param geom         AMReX geometry vector (one per level)
 *  \param include_bndry_corners  If true, include corner cells in flux sums
 *  \param terrain_blank  Optional terrain blanking field. Boundary locations
 *                        whose adjacent interior cell is terrain-blanked
 *                        (value 1) are excluded from area calculations and
 *                        velocity modifications.
 */
void enforceAdaptInflowSolvability(
    const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
        vels_vec,
    const amrex::GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const amrex::Vector<amrex::Geometry>& geom,
    bool include_bndry_corners = false,
    const IntField* terrain_blank = nullptr);

} // namespace kynema_sgf

#endif /* ADAPT_INFLOW_SOLVABILITY_H */

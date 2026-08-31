#ifndef ADAPT_INFLOW_SOLVABILITY_H
#define ADAPT_INFLOW_SOLVABILITY_H

#include "src/incflo_enums.H"
#include "AMReX_Array.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"

namespace kynema_sgf {

/** Enforce mass conservation for the adapt_inflow boundary condition.
 *
 *  Classifies each adapt_inflow boundary cell into one of three categories:
 *    - inflow:  velocity at the boundary points into the domain
 *    - outflow: velocity just inside the domain points outward
 *    - passive: neither inflow nor outflow
 *
 *  If total outflux exceeds total influx, a uniform inward velocity is applied
 *  to all passive cells to restore mass balance.
 *
 *  \param vels_vec     Per-level, per-direction normal velocity MultiFabs
 *  \param bc_types     Incflo BC type for each orientation
 *  \param geom         AMReX geometry vector (one per level)
 *  \param include_bndry_corners  If true, include corner cells in flux sums
 */
void enforceAdaptInflowSolvability(
    const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
        vels_vec,
    const amrex::GpuArray<BC, AMREX_SPACEDIM * 2>& bc_types,
    const amrex::Vector<amrex::Geometry>& geom,
    bool include_bndry_corners = false);

} // namespace kynema_sgf

#endif /* ADAPT_INFLOW_SOLVABILITY_H */

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
 *  Classifies each adapt_inflow boundary cell into one of three categories:
 *    - inflow:  velocity at the boundary points into the domain
 *    - outflow: velocity just inside the domain points outward
 *    - passive: neither inflow nor outflow
 *
 *  In addition to the scalar influx/outflux totals, the net (signed) outflux
 *  vector is computed to determine the direction the outflow is moving in.
 *  If outflux exceeds influx, a uniform inward velocity is applied only to
 *  passive cells on boundaries upstream of the outflux vector (e.g. xlo if
 *  the vector has a positive x component). If influx exceeds outflux, a
 *  uniform outward velocity is applied only to passive cells on boundaries
 *  downstream of the outflux vector instead.
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

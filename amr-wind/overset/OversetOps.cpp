#include "amr-wind/overset/OversetOps.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {

OversetOps::OversetOps(CFDSim& sim) : m_sim(sim)
{
    // Queries for reinitialization options
    amrex::ParmParse pp("Overset");
    pp.query("reinit_iterations", m_niterations);
    pp.query("reinit_convg_interval", m_calcconvint);
    pp.query("reinit_convg_tolerance", m_tol);
    pp.query("reinit_rlscale", m_niterations);
    pp.query("reinit_upw_margin", m_margin);
    pp.query("reinit_target_cutoff", m_target_cutoff);

    // Queries for coupling options
    pp.query("disable_coupled_nodal_proj", m_disable_nodal_proj);
    pp.query("disable_coupled_mac_proj", m_disable_mac_proj);
}

void OversetOps::post_init_actions()
{
    // Put vof check here
    m_vof_exists = m_sim.repo().field_exists("vof");
    ;

    // Output parameters if verbose
    parameter_output();
}

void OversetOps::parameter_output()
{
    // Print the details
    if (m_verbose > 0) {
        // Important parameters
        amrex::Print() << "Overset Coupling Parameters: \n"
                       << "---- Coupled nodal projection : "
                       << !m_disable_nodal_proj << std::endl
                       << "---- Coupled MAC projection   : "
                       << !m_disable_mac_proj << std::endl
                       << "---- Replace overset pres grad: " << m_replace_gp
                       << std::endl;
        if (m_vof_exists) {
            amrex::Print() << "Overset Reinitialization Parameters:\n"
                           << "---- Maximum iterations   : " << m_niterations
                           << std::endl
                           << "---- Convergence tolerance: " << m_tol
                           << std::endl
                           << "---- Relative length scale: " << m_rlscale
                           << std::endl
                           << "---- Upwinding VOF margin : " << m_margin
                           << std::endl;
        }
    }
    if (m_verbose > 1 && m_vof_exists) {
        // Less important or less used parameters
        amrex::Print() << "---- Calc. conv. interval : " << m_calcconvint
                       << std::endl
                       << "---- Target field cutoff  : " << m_target_cutoff
                       << std::endl;
    }
}

void OversetOps::pre_advance_actions()
{
    // Reinitialize fields
}

void OversetOps::post_advance_actions()
{
    // Replace and reapply pressure gradient if requested
}

} // namespace amr_wind
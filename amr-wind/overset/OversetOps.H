#ifndef OVERSETOPS_H
#define OVERSETOPS_H

#include "amr-wind/CFDSim.H"

namespace amr_wind {

class OversetOps
{
public:
    explicit OversetOps(CFDSim& sim);

    void post_init_actions();
    void pre_advance_actions();
    void post_advance_actions();

private:
    // Functions called within public functions
    void parameter_output();
    void sharpen_nalu_data();
    void replace_masked_gradp();

    // Check for multiphase sim
    bool m_vof_exists{false};

    // This is the only option for now
    const bool m_use_hs_pgrad{true};

    // Coupling options
    bool m_disable_nodal_proj{false};
    bool m_disable_mac_proj{false};
    bool m_replace_gp{false};

    // Verbosity
    int m_verbose{0};

    // Reinitialization parameters
    int m_niterations{10};
    int m_calcconvint{1}; // calctolniter, cconv
    amrex::Real m_tol = 1e-12;
    amrex::Real m_rlscale = 1.5;
    amrex::Real m_margin = 0.1;
    amrex::Real m_target_cutoff = 0.0; // proc_tgvof_tol

    CFDSim& m_sim;
    // Should I include sim, or should I have all of the fields here?
};

} // namespace amr_wind

#endif /* OVERSETOPS_H */
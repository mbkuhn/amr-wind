#include "amr-wind/incflo.H"

#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/utilities/tagging/RefinementCriteria.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/turbulence/TurbulenceModel.H"
#include "amr-wind/equation_systems/SchemeTraits.H"
#include "amr-wind/utilities/IOManager.H"
#include "amr-wind/utilities/PostProcessing.H"
#include "amr-wind/overset/OversetManager.H"

#include "AMReX_ParmParse.H"

using namespace amrex;

incflo::incflo()
    : m_sim(*this)
    , m_time(m_sim.time())
    , m_repo(m_sim.repo())
    , m_mesh_refiner(new amr_wind::RefineCriteriaManager(m_sim))
{
    // NOTE: Geometry on all levels has just been defined in the AmrCore
    // constructor. No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    m_time.parse_parameters();
    // Read inputs file using ParmParse
    ReadParameters();

    init_physics_and_pde();
}

incflo::~incflo() = default;

/** Initialize AMR mesh data structure.
 *
 *  Calls the AmrMesh/AmrCore functions to initialize the mesh. For restarted
 *  simulations, it loads the checkpoint file.
 */
void incflo::init_mesh()
{
    BL_PROFILE("amr-wind::incflo::init_mesh");
    // Initialize I/O manager to enable restart and outputs
    auto& io_mgr = m_sim.io_manager();
    io_mgr.initialize_io();
    for (auto& pp : m_sim.physics()) {
        pp->pre_init_actions();
    }

    if (!io_mgr.is_restart()) {
        // This tells the AmrMesh class not to iterate when creating the initial
        // grid hierarchy
        // SetIterateToFalse();

        // This tells the Cluster routine to use the new chopping routine which
        // rejects cuts if they don't improve the efficiency
        SetUseNewChop();

        // This is an AmrCore member function which recursively makes new levels
        // with MakeNewLevelFromScratch.
        amrex::Print() << "Creating mesh... ";
        InitFromScratch(m_time.current_time());
        amrex::Print() << "done" << std::endl;
        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Grid summary: " << std::endl;
            printGridSummary(amrex::OutStream(), 0, finest_level);
        }
    } else {
        // Read starting configuration from chk file.
        ReadCheckpointFile();
        for (int lev = finestLevel(); lev <= maxLevel(); ++lev) {
            regrid(lev, m_time.current_time());
        }
        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Grid summary: " << std::endl;
            printGridSummary(amrex::OutStream(), 0, finest_level);
        }
    }
}

/** Initialize AMR-Wind data structures after mesh has been created.
 *
 *  Modules initialized:
 *    - Registered \ref physics classes
 *    - Registered \ref PDE systems
 *    - Registered post-processing classes
 */
void incflo::init_amr_wind_modules()
{
    BL_PROFILE("amr-wind::incflo::init_amr_wind_modules");
    if (m_sim.has_overset()) {
        m_sim.overset_manager()->post_init_actions();
        m_ovst_ops.initialize(m_sim);
    } else {
        auto& mask_cell = m_sim.repo().declare_int_field("mask_cell", 1, 1);
        auto& mask_node = m_sim.repo().declare_int_field(
            "mask_node", 1, 1, 1, amr_wind::FieldLoc::NODE);
        mask_cell.setVal(1);
        mask_node.setVal(1);
    }

    for (auto& pp : m_sim.physics()) {
        pp->post_init_actions();
    }

    icns().initialize();
    for (auto& eqn : scalar_eqns()) {
        eqn->initialize();
    }
    // Initialize pressure gradient
    // (for cases with initial pressure field)
    UpdateGradP(
        sim().repo().get_field("density").vec_const_ptrs(),
        m_time.current_time(), m_time.delta_t());

    m_sim.pde_manager().fillpatch_state_fields(m_time.current_time());
    m_sim.post_manager().post_init_actions();
}

/** Initialize flow-field before performing time-integration.
 *
 *  This method calls the incflo::InitialProjection step to ensure that the
 *  velocity field is divergence free. Then it performs a user-defined number of
 *  initial iterations to compute \f$p^{n - 1/2}\f$ necessary for advancing into
 *  the first timestep. These actions are only performed when starting a new
 *  simulation, but not for a restarted simulation.
 */
void incflo::prepare_for_time_integration()
{
    BL_PROFILE("amr-wind::incflo::prepare_for_time_integration");
    // Don't perform initial work if this is a restart
    if (m_sim.io_manager().is_restart()) {
        return;
    }

    if (m_do_initial_proj) {
        InitialProjection();
    }
    if (m_initial_iterations > 0) {
        InitialIterations();
    }

    // Plot initial distribution
    if (m_time.write_plot_file()) {
        m_sim.io_manager().write_plot_file();
    }
    if (m_time.write_checkpoint()) {
        m_sim.io_manager().write_checkpoint_file();
    }
}

/** Perform all initialization actions for AMR-Wind.
 *
 *  This is a wrapper method that calls the following methods to perform the
 * actual work.
 *
 *  \callgraph
 */
void incflo::InitData()
{
    BL_PROFILE("amr-wind::incflo::InitData()");

    init_mesh();
    init_amr_wind_modules();
    prepare_for_time_integration();
}

/** Perform regrid actions at a given timestep.
 *
 *  \return Flag indicating if regrid was performed
 */
bool incflo::regrid_and_update()
{
    BL_PROFILE("amr-wind::incflo::regrid_and_update");

    if (m_time.do_regrid()) {
        amrex::Print() << "Regrid mesh ... ";
        amrex::Real rstart = amrex::ParallelDescriptor::second();
        regrid(0, m_time.current_time());
        amrex::Real rend = amrex::ParallelDescriptor::second() - rstart;
        amrex::Print() << "time elapsed = " << rend << std::endl;
        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Grid summary: " << std::endl;
            printGridSummary(amrex::OutStream(), 0, finest_level);
        }

        // update mesh map
        {
            if (m_sim.has_mesh_mapping()) {
                // TODO: Is this the only change required in presence of regrid
                // ?
                amrex::Print() << "Creating mesh mapping after regrid ... ";

                for (int lev = 0; lev <= finest_level; lev++) {
                    m_sim.mesh_mapping()->create_map(lev, Geom(lev));
                }
                amrex::Print() << "done" << std::endl;
            }
        }

        if (m_sim.has_overset()) {
            m_sim.overset_manager()->post_regrid_actions();
        } else {
            auto& mask_cell = m_sim.repo().get_int_field("mask_cell");
            auto& mask_node = m_sim.repo().get_int_field("mask_node");
            mask_cell.setVal(1);
            mask_node.setVal(1);
        }

        m_sim.pde_manager().fillpatch_state_fields(m_time.current_time());

        icns().post_regrid_actions();
        for (auto& eqn : scalar_eqns()) {
            eqn->post_regrid_actions();
        }
        for (auto& pp : m_sim.physics()) {
            pp->post_regrid_actions();
        }
        m_sim.post_manager().post_regrid_actions();
    }

    // update cell counts if uninitialized or if a regrid happened
    if (m_cell_count == -1 || m_time.do_regrid()) {
        m_cell_count = 0;
        for (int i = 0; i <= finest_level; i++) {
            m_cell_count += boxArray(i).numPts();
        }
    }

    return m_time.do_regrid();
}

/** Perform actions after a timestep
 *
 *  Performs any necessary I/O before advancing to next timestep
 */
void incflo::post_advance_work()
{
    BL_PROFILE("amr-wind::incflo::post_advance_work");

    m_sim.turbulence_model().post_advance_work();

    for (auto& pp : m_sim.physics()) {
        pp->post_advance_work();
    }

    m_sim.post_manager().post_advance_work();
    if (m_verbose > 1) {
        PrintMaxValues("end of timestep");
    }

    if (m_time.write_plot_file()) {
        m_sim.io_manager().write_plot_file();
    }

    if (m_time.write_checkpoint()) {
        m_sim.io_manager().write_checkpoint_file();
    }
}

/** Perform time-integration for user-defined time or timesteps.
 *
 *  \callgraph
 */
void incflo::Evolve()
{
    BL_PROFILE("amr-wind::incflo::Evolve()");

    while (m_time.new_timestep()) {
        amrex::Real time0 = amrex::ParallelDescriptor::second();

        regrid_and_update();

        if (m_prescribe_vel) {
            pre_advance_stage2();
            ComputePrescribeDt();
        } else {
            pre_advance_stage1();
            pre_advance_stage2();
        }

        amrex::Real time1 = amrex::ParallelDescriptor::second();
        // Advance to time t + dt
        do_advance();

        amrex::Print() << std::endl;
        amrex::Real time2 = amrex::ParallelDescriptor::second();
        post_advance_work();
        amrex::Real time3 = amrex::ParallelDescriptor::second();

        amrex::Print() << "WallClockTime: " << m_time.time_index()
                       << " Pre: " << std::setprecision(3) << (time1 - time0)
                       << " Solve: " << std::setprecision(4) << (time2 - time1)
                       << " Post: " << std::setprecision(3) << (time3 - time2)
                       << " Total: " << std::setprecision(4) << (time3 - time0)
                       << std::endl;

        amrex::Print() << "Solve time per cell: " << std::setprecision(4)
                       << amrex::ParallelDescriptor::NProcs() *
                              (time2 - time1) /
                              static_cast<amrex::Real>(m_cell_count)
                       << std::endl;
    }
    amrex::Print() << "\n======================================================"
                      "========================\n"
                   << std::endl;

    // Output at final time
    if (m_time.write_last_plot_file()) {
        m_sim.io_manager().write_plot_file();
    }
    if (m_time.write_last_checkpoint()) {
        m_sim.io_manager().write_checkpoint_file();
    }
}

void incflo::do_advance()
{
    /*
    if (m_sim.has_overset()) {
        m_ovst_ops.pre_advance_work();
    }
    */
    if (m_prescribe_vel) {
        prescribe_advance();
    } else {
        advance();
    }
    if (m_sim.has_overset()) {
        m_ovst_ops.post_advance_work();
    }
}

// Make a new level from scratch using provided BoxArray and
// DistributionMapping. Only used during initialization. overrides the pure
// virtual function in AmrCore
void incflo::MakeNewLevelFromScratch(
    int lev,
    Real time,
    const BoxArray& new_grids,
    const DistributionMapping& new_dmap)
{
    BL_PROFILE("amr-wind::incflo::MakeNewLevelFromScratch()");

    if (m_verbose > 0) {
        amrex::Print() << "Making new level " << lev << " from scratch"
                       << std::endl;
        if (m_verbose > 2) {
            amrex::Print() << "with BoxArray " << new_grids << std::endl;
        }
    }

    SetBoxArray(lev, new_grids);
    SetDistributionMap(lev, new_dmap);

    m_repo.make_new_level_from_scratch(lev, time, new_grids, new_dmap);

    // initialize the mesh map before initializing physics
    if (m_sim.has_mesh_mapping()) {
        m_sim.mesh_mapping()->create_map(lev, Geom(lev));
    }

    for (auto& pp : m_sim.physics()) {
        pp->initialize_fields(lev, Geom(lev));
    }
}

void incflo::init_physics_and_pde()
{
    // Check for mesh mapping
    m_sim.activate_mesh_map();

    {
        // Query and activate overset before initializing PDEs and physics
        amrex::ParmParse pp("incflo");
        bool activate_overset = false;
        pp.query("activate_overset", activate_overset);

        if (activate_overset) {
            m_sim.activate_overset();
        }
    }

    auto& pde_mgr = m_sim.pde_manager();

    // Always register incompressible Navier-Stokes equation
    pde_mgr.register_icns();

    // Register density first so that we can compute its `n+1/2` state before
    // other scalars attempt to use it in their computations.
    if (!pde_mgr.constant_density()) {
        if (!pde_mgr.scalar_eqns().empty()) {
            amrex::Abort(
                "For non-constant density, it must be the first equation "
                "registered for the scalar equations");
        }
        pde_mgr.register_transport_pde("Density");
    }

    m_sim.init_physics();
    {
        // Check for if velocity is prescribed
        amrex::ParmParse pp("incflo");
        pp.query("prescribe_velocity", m_prescribe_vel);

        // Check for sharpening iterations (overset feature)
        pp.query("sharpen_iterations", m_sharpen_iterations);
        pp.query("sharpen_tolerance", m_sharpen_tolerance);
        pp.query("sharpen_calctol_niter", m_sharpen_calctolniter);
        pp.query("sharpen_rlscale", m_sharpen_rlscale);
        pp.query("sharpen_margin", m_sharpen_margin);
        pp.query("sharpen_targetvof_tol", m_sharpen_proctg_tol);

        pp.query("sharpen_guess_hsp", m_sharpen_hsp_guess);
        pp.query("sharpen_replace_hsp", m_sharpen_hsp_replace);
        pp.query("sharpen_replace_gp", m_sharpen_replace_gp);
        pp.query("sharpen_pressure", m_sharpen_pressure);

        // Determine if overset values should be forced into projection
        pp.query("disable_overset_nodal", m_disable_onodal);
    }
    m_sim.create_turbulence_model();

    // Initialize the refinement criteria
    m_mesh_refiner->initialize();

    // Post-processing actions that need to declare fields
    m_sim.post_manager().pre_init_actions();
}

void incflo::ErrorEst(int lev, TagBoxArray& tags, Real time, int ngrow)
{
    BL_PROFILE("amr-wind::incflo::ErrorEst()");
    m_mesh_refiner->tag_cells(lev, tags, time, ngrow);
}

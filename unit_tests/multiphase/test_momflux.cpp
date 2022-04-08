#include "aw_test_utils/MeshTest.H"
#include "aw_test_utils/iter_tools.H"
#include "aw_test_utils/test_utils.H"
#include "amr-wind/physics/multiphase/MultiPhase.H"

namespace amr_wind_tests {

static void init_field3(
    amr_wind::Field& fld,
    const amrex::Real in0,
    const amrex::Real in1,
    const amrex::Real in2)
{
    run_algorithm(fld, [&](const int lev, const amrex::MFIter& mfi) {
        auto farr = fld(lev).array(mfi);
        const auto bx = mfi.growntilebox();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            farr(i, j, k, 0) = in0;
            farr(i, j, k, 1) = in1;
            farr(i, j, k, 2) = in2;
        });
    });
}

static void initialize_volume_fractions(
    const int dir,
    amr_wind::Field& vof)
{
    run_algorithm(vof, [&](const int lev, const amrex::MFIter& mfi) {
        auto vof_arr = vof(lev).array(mfi);
        const auto& bx = mfi.validbox();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int icheck;
            switch (dir) {
            case 0:
                icheck = i;
                break;
            case 1:
                icheck = j;
                break;
            case 2:
                icheck = k;
                break;
            }
            if (icheck > 0) {
                vof_arr(i, j, k) = 0.0;
            } else {
                vof_arr(i, j, k) = 1.0;
            }
        });
        // Left half is liquid, right half is gas
    });
}

static void check_results(
    const int dir,
    const amrex::Real rho1,
    const amrex::Real rho2,
    const amrex::Real vmag,
    const amrex::GpuArray<amrex::Real, 3> varr,
    amr_wind::Field& umac,
    amr_wind::Field& vmac,
    amr_wind::Field& wmac,
    amr_wind::Field& advalpha_f,
    amr_wind::Field& velocity,
    amr_wind::Field& conv_term)
{
    constexpr double tol = 1.0e-15;

    run_algorithm(velocity, [&](const int lev, const amrex::MFIter& mfi) {
        const auto& bx = mfi.validbox();

        const auto& um = umac(lev).array(mfi);
        const auto& vm = vmac(lev).array(mfi);
        const auto& wm = wmac(lev).array(mfi);
        const auto& af = advalpha_f(lev).array(mfi);
        const auto& vel = velocity(lev).array(mfi);
        const auto& dqdt = conv_term(lev).array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                int icheck;
                switch (dir) {
                case 0:
                    icheck = i;
                    break;
                case 1:
                    icheck = j;
                    break;
                case 2:
                    icheck = k;
                    break;
                }

                // Umac check is to make sure test is set up properly
                // vel check is to make sure mass-consistent fluxes worked

                // UMAC
                EXPECT_NEAR(um(i, j, k), varr[0], tol);
                // U
                EXPECT_NEAR(vel(i, j, k, 0), varr[0], tol);

                // VMAC
                EXPECT_NEAR(vm(i, j, k), varr[1], tol);
                // V
                EXPECT_NEAR(vel(i, j, k, 1), varr[1], tol);

                // WMAC
                EXPECT_NEAR(wm(i, j, k), varr[2], tol);
                // W
                EXPECT_NEAR(vel(i, j, k, 2), varr[2], tol);

                // Test volume fractions at faces
                if (icheck > 0) {
                    // Center face (coming from left cell)
                    EXPECT_NEAR(af(i, j, k), 1.0, tol);
                } else { // icheck == 0
                    // Left face (coming from right cell, periodic BC)
                    EXPECT_NEAR(af(i, j, k), 0.0, tol);
                }

                // Test momentum fluxes by checking convective term
                if (icheck > 0) {
                    // Right cell (liquid entering, gas leaving)
                    EXPECT_NEAR(
                        dqdt(i, j, k, dir), vmag * vmag * (rho1 - rho2) / 0.5,
                        tol);
                } else { // icheck == 0
                    // Left cell (gas entering, liquid leaving)
                    EXPECT_NEAR(
                        dqdt(i, j, k, dir), vmag * vmag * (rho2 - rho1) / 0.5,
                        tol);
                }
            });
    });
}

class MassMomFluxOpTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{2, 2, 2}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", 2);
            pp.addarr("n_cell", ncell);
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0, 0.0, 0.0}};
            amrex::Vector<amrex::Real> probhi{{1.0, 1.0, 1.0}};

            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
        }
        {
            amrex::ParmParse pp("MultiPhase");
            pp.add("density_fluid1", m_rho1);
            pp.add("density_fluid2", m_rho2);
        }
        {
            amrex::ParmParse pp("incflo");
            amrex::Vector<std::string> physics{"MultiPhase"};
            pp.addarr("physics", physics);
            pp.add("use_godunov", (int)1);
            pp.add("godunov_type", (std::string) "weno");
        }
        {
            amrex::ParmParse pp("time");
            pp.add("fixed_dt", dt);
        }
        {
            amrex::ParmParse pp("transport");
            pp.add("model", (std::string) "TwoPhaseTransport");
            pp.add("viscosity_fluid1", (amrex::Real)0.0);
            pp.add("viscosity_fluid2", (amrex::Real)0.0);
        }
    }

    void testing_coorddir(const int dir)
    {

        populate_parameters();
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<int> periodic{{1, 1, 1}};
            pp.addarr("is_periodic", periodic);
        }

        initialize_mesh();

        auto& repo = sim().repo();

        // Set up icns PDE, access to VOF PDE
        auto& pde_mgr = sim().pde_manager();
        auto& mom_eqn = pde_mgr.register_icns();
        mom_eqn.initialize();

        // Initialize physics for the sake of MultiPhase routines
        sim().init_physics();

        // Initialize constant velocity field
        amrex::GpuArray<amrex::Real, 3> varr = {0};
        varr[dir] = m_vel;
        auto& velocity = mom_eqn.fields().field;
        init_field3(velocity, varr[0], varr[1], varr[2]);

        // Initialize volume fraction field
        auto& vof = repo.get_field("vof");
        initialize_volume_fractions(dir, vof);
        // Populate boundary cells
        vof.fillpatch(0.0);
        // Sync density field with vof
        auto& mphase = sim().physics_manager().get<amr_wind::MultiPhase>();
        mphase.set_density_via_vof();

        // Advance states (new -> old)
        sim().pde_manager().advance_states();

        // Perform pre-advection step to get MAC velocity field (should be
        // uniform)
        mom_eqn.pre_advection_actions(amr_wind::FieldState::Old);

        // Perform VOF solve
        for (auto& seqn : pde_mgr.scalar_eqns()) {
            if (seqn->fields().field.base_name() == "vof") {
                seqn->initialize();
                seqn->compute_advection_term(amr_wind::FieldState::Old);
                seqn->post_solve_actions();
            }
        }

        // Zero unused momentum terms: src (pressure)
        auto& grad_p = repo.get_field("gp");
        init_field3(grad_p, 0.0, 0.0, 0.0);
        // Setup mask_cell array to avoid errors in solve
        auto& mask_cell = repo.declare_int_field("mask_cell", 1, 1);
        mask_cell.setVal(1);

        // Perform momentum solve
        mom_eqn.compute_advection_term(amr_wind::FieldState::Old);
        mom_eqn.compute_diffusion_term(amr_wind::FieldState::New);
        mom_eqn.compute_source_term(amr_wind::FieldState::New);
        mom_eqn.compute_predictor_rhs(DiffusionType::Explicit);

        // Get MAC velocities for use in testing
        auto& umac = repo.get_field("u_mac");
        auto& vmac = repo.get_field("v_mac");
        auto& wmac = repo.get_field("w_mac");

        // Get advected alpha
        std::string sdir = "d";
        switch (dir) {
        case 0:
            sdir = "x";
            break;
        case 1:
            sdir = "y";
            break;
        case 2:
            sdir = "z";
            break;
        }
        auto& advalpha_f = repo.get_field("advalpha_" + sdir);

        // Get convective term
        auto& conv_term =
            mom_eqn.fields().conv_term.state(amr_wind::FieldState::New);

        // Perform checks after single time step has advanced
        check_results(
            dir, m_rho1, m_rho2, m_vel, varr, umac, vmac, wmac, advalpha_f,
            velocity, conv_term);
    }
    const amrex::Real m_rho1 = 1000.0;
    const amrex::Real m_rho2 = 1.0;
    const amrex::Real m_vel = 5.0;
    const amrex::Real dt = 0.45 * 0.5 / m_vel; // first number is CFL
};

TEST_F(MassMomFluxOpTest, fluxfaceX) { testing_coorddir(0); }
TEST_F(MassMomFluxOpTest, fluxfaceY) { testing_coorddir(1); }
TEST_F(MassMomFluxOpTest, fluxfaceZ) { testing_coorddir(2); }

} // namespace amr_wind_tests

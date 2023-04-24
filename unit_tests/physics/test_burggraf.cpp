#include "aw_test_utils/MeshTest.H"
#include "amr-wind/equation_systems/icns/source_terms/BurggrafFlowForcing.H"
#include "aw_test_utils/iter_tools.H"

namespace amr_wind_tests {

class BurggrafFlowTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{nx, ny, nz}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", nz);
            pp.addarr("n_cell", ncell);
        }
        {
            amrex::ParmParse pp("geometry");
            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
        }
        {
            amrex::ParmParse pp("transport");
            pp.add("viscosity", mu);
        }
        {
            amrex::ParmParse pp("incflo");
            amrex::Vector<std::string> physics{"BurggrafFlow"};
            pp.addarr("physics", physics);
        }
        {
            amrex::ParmParse pp("ICNS");
            amrex::Vector<std::string> src_terms{"BurggrafFlowForcing"};
            pp.addarr("source_terms", src_terms);
        }
        {
            // Periodicity
            amrex::ParmParse pp("geometry");
            amrex::Vector<int> periodic{{0, 0, 1}};
            pp.addarr("is_periodic", periodic);
            // Boundary conditions
            amrex::ParmParse ppxlo("xlo");
            ppxlo.add("type", (std::string) "no_slip_wall");
            amrex::ParmParse ppylo("ylo");
            ppylo.add("type", (std::string) "no_slip_wall");
            amrex::ParmParse ppxhi("xhi");
            ppxhi.add("type", (std::string) "no_slip_wall");
            amrex::ParmParse ppyhi("yhi");
            ppyhi.add("type", (std::string) "mass_inflow");
            ppyhi.add("density", 1.0);
            ppyhi.add("velocity.inflow_type", (std::string) "BurggrafLid");
        }
    }
    // Parameters
    const amrex::Vector<amrex::Real> problo{{0.0, 0.0, 0.0}};
    const amrex::Vector<amrex::Real> probhi{{1.0, 1.0, 0.25}};
    const int nx = 8;
    const int ny = 8;
    const int nz = 2;
    const amrex::Real mu = 1e-2;
    const amrex::Real tol = 1e-12;
};

namespace {
amrex::Real get_val_at_xy(
    amr_wind::Field& field,
    const int lev,
    const int comp,
    const amrex::Real plox,
    const amrex::Real dx,
    const amrex::Real ploy,
    const amrex::Real dy,
    const amrex::Real xtarg,
    const amrex::Real ytarg)
{
    amrex::Real error_total = 0;

    error_total += amrex::ReduceSum(
        field(lev), 0,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx,
            amrex::Array4<amrex::Real const> const& f_arr) -> amrex::Real {
            amrex::Real error = 0;

            amrex::Loop(bx, [=, &error](int i, int j, int k) noexcept {
                const amrex::Real x = plox + (0.5 + i) * dx;
                const amrex::Real y = ploy + (0.5 + j) * dy;
                // Check if current cell is closest to desired height
                if ((std::abs(x - xtarg) < 0.5 * dx) &&
                    (std::abs(y - ytarg) < 0.5 * dy)) {
                    // Add field value to output
                    error += f_arr(i, j, k, comp);
                }
            });

            return error;
        });
    amrex::ParallelDescriptor::ReduceRealSum(error_total);
    return error_total;
}
} // namespace

TEST_F(BurggrafFlowTest, src_vals)
{
    constexpr amrex::Real tol = 1.0e-12;
    populate_parameters();
    initialize_mesh();

    auto& pde_mgr = sim().pde_manager();
    pde_mgr.register_icns();
    sim().init_physics();

    // Do initialization of fields through physics
    const int lev = 0;
    for (auto& pp : sim().physics()) {
        pp->pre_init_actions();
        pp->initialize_fields(lev, sim().mesh().Geom(lev));
    }

    // Get fields that will be tested
    auto& src_term = pde_mgr.icns().fields().src_term;
    auto& bf_field = sim().repo().get_field("bf_src_term");

    // Perform source term operation
    amr_wind::pde::icns::BurggrafFlowForcing burg_src(sim());
    src_term.setVal(0.0);
    run_algorithm(src_term, [&](const int lev, const amrex::MFIter& mfi) {
        const auto& bx = mfi.tilebox();
        const auto& src_arr = src_term(lev).array(mfi);

        burg_src(lev, mfi, bx, amr_wind::FieldState::New, src_arr);
    });

    const amrex::Real dx = sim().mesh().Geom(0).CellSizeArray()[0];
    const amrex::Real plox = sim().mesh().Geom(0).ProbLoArray()[0];
    const amrex::Real dy = sim().mesh().Geom(0).CellSizeArray()[0];
    const amrex::Real ploy = sim().mesh().Geom(0).ProbLoArray()[0];

    // Target x and y values for check
    amrex::Real x = 0.4375;
    amrex::Real y = 0.3125;

    // Check values in bf_field, which is part of initialization
    auto vals_intz = get_val_at_xy(bf_field, 0, 1, plox, dx, ploy, dy, x, y);
    EXPECT_NEAR(vals_intz / nz, 0.07500815872, tol);

    // Check values in source term, which is a copy of bf_field
    vals_intz = get_val_at_xy(src_term, 0, 1, plox, dx, ploy, dy, x, y);
    EXPECT_NEAR(vals_intz / nz, 0.07500815872, tol);
}

} // namespace amr_wind_tests

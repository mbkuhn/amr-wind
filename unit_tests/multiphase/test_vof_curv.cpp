#include "aw_test_utils/MeshTest.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/equation_systems/vof/curvature.H"
#include "aw_test_utils/test_utils.H"
#include "aw_test_utils/iter_tools.H"

namespace amr_wind_tests {

namespace {
void initialize_levelset(
    amr_wind::Field& levelset,
    amrex::Vector<amrex::Geometry> geom,
    const amrex::Real curv)
{

    run_algorithm(levelset, [&](const int lev, const amrex::MFIter& mfi) {
        auto lvs_arr = levelset(lev).array(mfi);
        const auto& gbx = mfi.growntilebox(1);
        const auto& dx = geom[lev].CellSizeArray();
        const auto& problo = geom[lev].ProbLoArray();
        // Negative radius indicates plane at x = 0.5625
        const amrex::Real radius = (curv > 0.0) ? 2.0 / curv : -1.0;
        // Interface is intended to be at x = 0.5625; y = 0.5625; z = 0.5625
        const amrex::Real yc = 0.5625;
        const amrex::Real zc = 0.5625;
        // Center is moved according to radius
        const amrex::Real xc = 0.5625 + amrex::max(radius, 0.0);
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
            const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
            const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
            lvs_arr(i, j, k) =
                (radius > 0.0)
                    ? (radius - std::sqrt(
                                    (x - xc) * (x - xc) + (y - yc) * (y - yc) +
                                    (z - zc) * (z - zc)))
                    : (x - xc);
        });
    });
}
void initialize_volume_fractions(
    amr_wind::Field& vof,
    amr_wind::Field& levelset,
    amrex::Vector<amrex::Geometry> geom)
{

    run_algorithm(vof, [&](const int lev, const amrex::MFIter& mfi) {
        auto vof_arr = vof(lev).array(mfi);
        auto lvs_arr = levelset(lev).array(mfi);
        const auto& vbx = mfi.validbox();
        const auto& dx = geom[lev].CellSizeArray();
        const amrex::Real eps = 2. * std::cbrt(dx[0] * dx[1] * dx[2]);
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                vof_arr(i, j, k) = amr_wind::multiphase::levelset_to_vof(
                    i, j, k, eps, lvs_arr);
            });
    });
}
} // namespace
class VOFCurvTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{m_nx, m_nx, m_nx}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", m_nx);
            pp.addarr("n_cell", ncell);
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0, 0.0, 0.0}};
            amrex::Vector<amrex::Real> probhi{{1.0, 1.0, 1.0}};

            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
        }
    }

    void testing_curvature(amrex::Real curv)
    {
        constexpr double tol = 1.0e-15;
        populate_parameters();

        initialize_mesh();

        auto& repo = sim().repo();

        // Initialize volume fraction field with levelset
        auto& vof = repo.declare_field("vof", 1, 0);
        auto& lvs = repo.declare_field("levelset", 1, 1);
        const auto& geom = repo.mesh().Geom();
        initialize_levelset(lvs, geom, curv);
        initialize_volume_fractions(vof, lvs, geom);

        // Calculate curvature at interface, center of domain
        amrex::Real mycurv = -100.0;
        const int lev = 0;
        const int sample_i = m_nx / 2;
        const int sample_j = m_nx / 2;
        const int sample_k = m_nx / 2;
        for (amrex::MFIter mfi(vof(lev)); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            if (vbx.contains(sample_i, sample_j, sample_k)) {
                mycurv = amr_wind::multiphase::paraboloid_fit(
                    sample_i, sample_j, sample_k, vof(lev).array(mfi),
                    geom[0].ProbLoArray(), geom[0].CellSizeArray());
                std::cout << (vof(lev).array(mfi))(sample_i,sample_j,sample_k) << std::endl;
            }
        }

        // Check curvature
        EXPECT_NEAR(mycurv, -curv, tol);
    }
    const int m_nx = 8;
};

TEST_F(VOFCurvTest, zero) { testing_curvature(0.0); }
TEST_F(VOFCurvTest, half) { testing_curvature(0.1); }
} // namespace amr_wind_tests
#include "aw_test_utils/MeshTest.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
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
        const auto& gbx3 = mfi.growntilebox(3);
        const auto& dx = geom[lev].CellSizeArray();
        const auto& problo = geom[lev].ProbLoArray();
        // Negative radius indicates plane at x = 0.5625
        const amrex::Real radius = (curv > 0.0) ? 2.0 / curv : -1.0;
        // Interface is intended to be at x = 0.5625; y = 0.5625; z = 0.5625
        const amrex::Real yc = 0.5625;
        const amrex::Real zc = 0.5625;
        // Center is moved according to radius
        const amrex::Real xc = 0.5625 + amrex::max(radius, 0.0);
        amrex::ParallelFor(gbx3, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
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
        const auto& gbx2 = mfi.growntilebox(2);
        const auto& dx = geom[lev].CellSizeArray();
        const amrex::Real eps = 2. * std::cbrt(dx[0] * dx[1] * dx[2]);
        amrex::ParallelFor(
            gbx2, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                vof_arr(i, j, k) = amr_wind::multiphase::levelset_to_vof(
                    i, j, k, eps, lvs_arr);
            });
    });
}
amrex::Real normal_vector_comp_impl(amr_wind::Field& vof)
{
    amrex::Real error_total = 0.0;

    for (int lev = 0; lev < vof.repo().num_active_levels(); ++lev) {

        error_total += amrex::ReduceMax(
            vof(lev), 0,
            [=] AMREX_GPU_HOST_DEVICE(
                amrex::Box const& bx,
                amrex::Array4<amrex::Real const> const& vof_arr)
                -> amrex::Real {
                amrex::Real error = 0.0;

                amrex::Loop(bx, [=, &error](int i, int j, int k) noexcept {
                    amrex::Real mx1, my1, mz1, mx2, my2, mz2;

                    // Check if cell contains intermediate VOF
                    if (vof_arr(i, j, k) < 1.0 - 1e-12 &&
                        vof_arr(i, j, k) > 1e-12) {
                        // Get MYC normal
                        amr_wind::multiphase::mixed_youngs_central_normal(
                            i, j, k, vof_arr, mx1, my1, mz1);
                        // Get Swartz normal
                        amr_wind::multiphase::swartz_onestep_normal(
                            i, j, k, vof_arr, mx2, my2, mz2);

                        // Use L_inf norm, normal results should be similar
                        error = amrex::max(error, amrex::Math::abs(mx1 - mx2));
                        error = amrex::max(error, amrex::Math::abs(my1 - my2));
                        error = amrex::max(error, amrex::Math::abs(mz1 - mz2));
                        // Check for nan
                        error = amrex::max(error, (mx2 != mx2) ? 1.0 : 0.0);
                        error = amrex::max(error, (my2 != my2) ? 1.0 : 0.0);
                        error = amrex::max(error, (mz2 != mz2) ? 1.0 : 0.0);
                    }
                });

                return error;
            });
    }
    return error_total;
}
} // namespace
class VOFNormTest : public MeshTest
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

    void testing_normals(amrex::Real curv)
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

        // Calculate and compare normal vector throughout
        amrex::Real error_total = normal_vector_comp_impl(vof);
        amrex::ParallelDescriptor::ReduceRealSum(error_total);
        EXPECT_NEAR(error_total, 0.0, tol);
    }
    const int m_nx = 8;
};

TEST_F(VOFNormTest, zero) { testing_normals(0.0); }
TEST_F(VOFNormTest, half) { testing_normals(0.1); }
} // namespace amr_wind_tests
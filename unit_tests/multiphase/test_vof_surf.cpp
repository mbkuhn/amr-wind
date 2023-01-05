#include "aw_test_utils/MeshTest.H"
#include "aw_test_utils/iter_tools.H"
#include "aw_test_utils/test_utils.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"

namespace amr_wind_tests {

class VOFSurfTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{4, 4, 4}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", 4);
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
};

/*namespace {

void initialize_volume_fractions(
    const int dir,
    const amrex::Box& bx,
    const amrex::Array4<amrex::Real>& vof_arr)
{
    // grow the box by 1 so that x,y,z go out of bounds and min(max()) corrects
    // it and it fills the ghosts with wall values
    const int d = dir;
    amrex::ParallelFor(grow(bx, 1), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        int ii = (d != 0 ? i : 0);
        int jj = (d != 1 ? j : 0);
        int kk = (d != 2 ? k : 0);
        if (ii + jj + kk > 3) {
            vof_arr(i, j, k) = 0.0;
        }
        if (ii + jj + kk == 3) {
            vof_arr(i, j, k) = 0.5;
        }
        if (ii + jj + kk < 3) {
            vof_arr(i, j, k) = 1.0;
        }
    });
}

void init_vof(amr_wind::Field& vof, const int dir)
{
    run_algorithm(vof, [&](const int lev, const amrex::MFIter& mfi) {
        auto vof_arr = vof(lev).array(mfi);
        const auto& bx = mfi.validbox();
        initialize_volume_fractions(dir, bx, vof_arr);
    });
}

void initialize_volume_fractions_horizontal(
    const int dir,
    const amrex::Box& bx,
    const amrex::Real vof_val,
    const amrex::Array4<amrex::Real>& vof_arr)
{
    // grow the box by 1 so that x,y,z go out of bounds and min(max()) corrects
    // it and it fills the ghosts with wall values
    const int d = dir;
    const amrex::Real vv = vof_val;
    amrex::ParallelFor(grow(bx, 1), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        int ii = (d == 0 ? i : (d == 1 ? j : k));
        if (ii > 1) {
            vof_arr(i, j, k) = 0.0;
        }
        if (ii == 1) {
            vof_arr(i, j, k) = vv;
        }
        if (ii < 1) {
            vof_arr(i, j, k) = 1.0;
        }
    });
}

void init_vof_h(amr_wind::Field& vof, const amrex::Real vof_val, const int dir)
{
    run_algorithm(vof, [&](const int lev, const amrex::MFIter& mfi) {
        auto vof_arr = vof(lev).array(mfi);
        const auto& bx = mfi.validbox();
        initialize_volume_fractions_horizontal(dir, bx, vof_val, vof_arr);
    });
}

amrex::Real normal_vector_test_impl(amr_wind::Field& vof, const int dir)
{
    amrex::Real error_total = 0.0;
    const int d = dir;

    for (int lev = 0; lev < vof.repo().num_active_levels(); ++lev) {

        error_total += amrex::ReduceSum(
            vof(lev), 0,
            [=] AMREX_GPU_HOST_DEVICE(
                amrex::Box const& bx,
                amrex::Array4<amrex::Real const> const& vof_arr)
                -> amrex::Real {
                amrex::Real error = 0.0;

                amrex::Loop(bx, [=, &error](int i, int j, int k) noexcept {
                    amrex::Real mx, my, mz;
                    amr_wind::multiphase::mixed_youngs_central_normal(
                        i, j, k, vof_arr, mx, my, mz);

                    int ii = (d != 0 ? i : 0);
                    int jj = (d != 1 ? j : 0);
                    int kk = (d != 2 ? k : 0);

                    // Use L1 norm, check cells where slope is known
                    if (ii + jj + kk == 3) {
                        error += amrex::Math::abs(mx - (d != 0 ? 0.5 : 0.0));
                        error += amrex::Math::abs(my - (d != 1 ? 0.5 : 0.0));
                        error += amrex::Math::abs(mz - (d != 2 ? 0.5 : 0.0));
                    }
                });

                return error;
            });
    }
    return error_total;
}

amrex::Real fit_plane_test_impl(amr_wind::Field& vof, const int dir)
{
    amrex::Real error_total = 0.0;
    const int d = dir;

    for (int lev = 0; lev < vof.repo().num_active_levels(); ++lev) {

        error_total += amrex::ReduceSum(
            vof(lev), 0,
            [=] AMREX_GPU_HOST_DEVICE(
                amrex::Box const& bx,
                amrex::Array4<amrex::Real const> const& vof_arr)
                -> amrex::Real {
                amrex::Real error = 0.0;

                amrex::Loop(bx, [=, &error](int i, int j, int k) noexcept {
                    int ii = (d != 0 ? i : 0);
                    int jj = (d != 1 ? j : 0);
                    int kk = (d != 2 ? k : 0);
                    // Check multiphase cells
                    if (ii + jj + kk == 3) {
                        amrex::Real mx, my, mz, alpha;
                        amr_wind::multiphase::fit_plane(
                            i, j, k, vof_arr, mx, my, mz, alpha);

                        // Check slope
                        error += amrex::Math::abs(mx - (d != 0 ? 0.5 : 0.0));
                        error += amrex::Math::abs(my - (d != 1 ? 0.5 : 0.0));
                        error += amrex::Math::abs(mz - (d != 2 ? 0.5 : 0.0));
                        // Check intercept
                        error += amrex::Math::abs(alpha - 0.5);
                    }
                });

                return error;
            });
    }
    return error_total;
}

amrex::Real fit_plane_test_impl_h(
    amr_wind::Field& vof, const amrex::Real vof_val, const int dir)
{
    amrex::Real error_total = 0.0;
    const int d = dir;
    const amrex::Real vv = vof_val;

    for (int lev = 0; lev < vof.repo().num_active_levels(); ++lev) {

        error_total += amrex::ReduceSum(
            vof(lev), 0,
            [=] AMREX_GPU_HOST_DEVICE(
                amrex::Box const& bx,
                amrex::Array4<amrex::Real const> const& vof_arr)
                -> amrex::Real {
                amrex::Real error = 0.0;

                amrex::Loop(bx, [=, &error](int i, int j, int k) noexcept {
                    int ii = (d == 0 ? i : (d == 1 ? j : k));
                    // Check multiphase cells
                    if (ii == 1) {
                        amrex::Real mx, my, mz, alpha;
                        amr_wind::multiphase::fit_plane(
                            i, j, k, vof_arr, mx, my, mz, alpha);

                        // Check slope
                        error += amrex::Math::abs(mx - (d == 0 ? 1.0 : 0.0));
                        error += amrex::Math::abs(my - (d == 1 ? 1.0 : 0.0));
                        error += amrex::Math::abs(mz - (d == 2 ? 1.0 : 0.0));
                        // Check intercept
                        error += amrex::Math::abs(alpha - vv);
                    }
                });

                return error;
            });
    }
    return error_total;
}

} // namespace*/

TEST_F(VOFSurfTest, simple_half)
{
    // Structured cell boundaries
    const amrex::Real xm = 0.0;
    const amrex::Real xp = 1.0;
    const amrex::Real ym = 0.0;
    const amrex::Real yp = 1.0;
    const amrex::Real zm = 0.0;
    const amrex::Real zp = 1.0;
    // Constant input quantity
    const amrex::Real VOF = 0.5;

    // Output quantities to be reused
    amrex::Real xc = 0.0;
    amrex::Real yc = 0.0;
    amrex::Real zc = 0.0;
    amrex::Real sa = 0.0;

    // Tolerance for checks
    constexpr amrex::Real tol = 1e-11;
    // In x
    {
        // Input quantities: normal vector
        const amrex::Real normx = 1.0;
        const amrex::Real normy = 0.0;
        const amrex::Real normz = 0.0;

        // Get alpha based on inputs
        const amrex::Real alpha =
            amr_wind::multiphase::volume_intercept(normx, normy, normz, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        // centroid should be center of cell
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        EXPECT_NEAR(zc, 0.5 * (zm + zp), tol);
        // surface area should be xy area
        EXPECT_NEAR(sa, (xp - xm) * (yp - ym), tol);
    }
    // In y
    {
        // Input quantities: normal vector
        const amrex::Real normx = 0.0;
        const amrex::Real normy = 1.0;
        const amrex::Real normz = 0.0;

        // Get alpha based on inputs
        const amrex::Real alpha =
            amr_wind::multiphase::volume_intercept(normx, normy, normz, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        // centroid should be center of cell
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        EXPECT_NEAR(zc, 0.5 * (zm + zp), tol);
        // surface area should be xy area
        EXPECT_NEAR(sa, (xp - xm) * (yp - ym), tol);
    }
    // In z
    {
        // Input quantities: normal vector
        const amrex::Real normx = 0.0;
        const amrex::Real normy = 0.0;
        const amrex::Real normz = 1.0;

        // Get alpha based on inputs
        const amrex::Real alpha =
            amr_wind::multiphase::volume_intercept(normx, normy, normz, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        // centroid should be center of cell
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        EXPECT_NEAR(zc, 0.5 * (zm + zp), tol);
        // surface area should be xy area
        EXPECT_NEAR(sa, (xp - xm) * (yp - ym), tol);
    }
}

TEST_F(VOFSurfTest, other_half)
{
    // Structured cell boundaries
    const amrex::Real xm = 0.1;
    const amrex::Real xp = 0.5;
    const amrex::Real ym = -0.2;
    const amrex::Real yp = 0.1;
    const amrex::Real zm = 0.0;
    const amrex::Real zp = 0.25;
    // Input quantities: normal vector and vof
    const amrex::Real normx = 0.0;
    const amrex::Real normy = 0.0;
    const amrex::Real normz = 1.0;
    const amrex::Real VOF = 0.5;

    // Get alpha based on inputs
    const amrex::Real alpha =
        amr_wind::multiphase::volume_intercept(normx, normy, normz, VOF);

    // Output quantities to be reused
    amrex::Real xc = 0.0;
    amrex::Real yc = 0.0;
    amrex::Real zc = 0.0;
    amrex::Real sa = 0.0;

    // Perform calculations
    amr_wind::multiphase::surfacearea_center(
        normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

    // Check answers
    constexpr amrex::Real tol = 1e-11;
    // centroid should be center of cell
    EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
    EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
    EXPECT_NEAR(zc, 0.5 * (zm + zp), tol);
    // surface area should be xy area
    EXPECT_NEAR(sa, (xp - xm) * (yp - ym), tol);
}

TEST_F(VOFSurfTest, select)
{
    // Structured cell boundaries
    const amrex::Real xm = 0.1;
    const amrex::Real xp = 0.5;
    const amrex::Real ym = -0.2;
    const amrex::Real yp = 0.1;
    const amrex::Real zm = 0.0;
    const amrex::Real zp = 0.25;
    // Output quantities to be reused
    amrex::Real xc = 0.0;
    amrex::Real yc = 0.0;
    amrex::Real zc = 0.0;
    amrex::Real sa = 0.0;

    // Horizontal plane, less than half VOF
    {
        // Input quantities: normal vector and vof
        const amrex::Real normx = 0.0;
        const amrex::Real normy = 0.0;
        const amrex::Real normz = 1.0;
        const amrex::Real VOF = 0.25;

        // Get alpha based on inputs
        const amrex::Real alpha =
            amr_wind::multiphase::volume_intercept(normx, normy, normz, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        constexpr amrex::Real tol = 1e-11;
        // centroid of should be center of cell in x, y
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        // 1/4 of cell in z
        EXPECT_NEAR(zc, zm + 0.25 * (zp - zm), tol);
        // surface area should be xy area
        EXPECT_NEAR(sa, (xp - xm) * (yp - ym), tol);
    }
    // Horizontal plane, more than half VOF
    {
        // Input quantities: normal vector and vof
        const amrex::Real normx = 0.0;
        const amrex::Real normy = 0.0;
        const amrex::Real normz = 1.0;
        const amrex::Real VOF = 0.75;

        // Get alpha based on inputs
        const amrex::Real alpha =
            amr_wind::multiphase::volume_intercept(normx, normy, normz, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        constexpr amrex::Real tol = 1e-11;
        // centroid should be center of cell in x, y
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        // 3/4 of cell in z
        EXPECT_NEAR(zc, zm + 0.75 * (zp - zm), tol);
        // surface area should be xy area
        EXPECT_NEAR(sa, (xp - xm) * (yp - ym), tol);
    }
    // Diagonal plane in 2D, half cell
    {
        // Input quantities: normal vector and vof
        const amrex::Real normx = (zp - zm);
        const amrex::Real normy = 0.0;
        const amrex::Real normz = (xp - xm);
        const amrex::Real VOF = 0.5;

        // Account for normalization and modify for unit cell
        const amrex::Real unitnormx = normx * (xp - xm);
        const amrex::Real unitnormy = normy * (yp - ym);
        const amrex::Real unitnormz = normz * (zp - zm);
        const amrex::Real nmag = unitnormx + unitnormy + unitnormz;

        // Get alpha based on inputs
        amrex::Real alpha = amr_wind::multiphase::volume_intercept(
            unitnormx / nmag, unitnormy / nmag, unitnormz / nmag, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        constexpr amrex::Real tol = 1e-11;
        // centroid should be center of cell in x, y, z
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        EXPECT_NEAR(zc, 0.5 * (zm + zp), tol);
        // surface area should be y length times xz hypotenuse
        EXPECT_NEAR(
            sa, sqrt(std::pow(xp - xm, 2) + std::pow(zp - zm, 2)) * (yp - ym),
            tol);
    }
    // Diagonal plane in 2D, eighth of cell
    {
        // Input quantities: normal vector and vof
        const amrex::Real normx = (zp - zm);
        const amrex::Real normy = 0.0;
        const amrex::Real normz = (xp - xm);
        const amrex::Real VOF = 0.125;

        // Account for normalization and modify for unit cell
        const amrex::Real unitnormx = normx * (xp - xm);
        const amrex::Real unitnormy = normy * (yp - ym);
        const amrex::Real unitnormz = normz * (zp - zm);
        const amrex::Real nmag = unitnormx + unitnormy + unitnormz;

        // Get alpha based on inputs
        amrex::Real alpha = amr_wind::multiphase::volume_intercept(
            unitnormx / nmag, unitnormy / nmag, unitnormz / nmag, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        constexpr amrex::Real tol = 1e-11;
        // centroid should be center of cell in y
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        // quarter of cell in x, z
        EXPECT_NEAR(xc, xm + 0.25 * (xp - xm), tol);
        EXPECT_NEAR(zc, zm + 0.25 * (zp - zm), tol);
        // surface area should be y length times xz hypotenuse
        EXPECT_NEAR(
            sa,
            sqrt(std::pow(0.5 * (xp - xm), 2) + std::pow(0.5 * (zp - zm), 2)) *
                (yp - ym),
            tol);
    }
    // Diagonal plane in 3D, half of cell
    {
        // Input quantities: normal vector and vof
        const amrex::Real normx = (yp - ym) * (zp - zm);
        const amrex::Real normy = (xp - xm) * (zp - zm);
        const amrex::Real normz = 2.0 * (xp - xm) * (yp - ym);
        const amrex::Real VOF = 0.5;

        // Account for normalization and modify for unit cell
        const amrex::Real unitnormx = normx * (xp - xm);
        const amrex::Real unitnormy = normy * (yp - ym);
        const amrex::Real unitnormz = normz * (zp - zm);
        const amrex::Real nmag = unitnormx + unitnormy + unitnormz;

        // Get alpha based on inputs
        amrex::Real alpha = amr_wind::multiphase::volume_intercept(
            unitnormx / nmag, unitnormy / nmag, unitnormz / nmag, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        constexpr amrex::Real tol = 1e-11;
        // centroid should be center of cell in x, y, z
        EXPECT_NEAR(xc, 0.5 * (xm + xp), tol);
        EXPECT_NEAR(yc, 0.5 * (ym + yp), tol);
        EXPECT_NEAR(zc, 0.5 * (zm + zp), tol);
        // analytical surface area
        const amrex::Real a =
            sqrt(std::pow((xp - xm), 2) + std::pow(0.5 * (zp - zm), 2));
        const amrex::Real b =
            sqrt(std::pow((yp - ym), 2) + std::pow(0.5 * (zp - zm), 2));
        const amrex::Real c = sqrt(
            std::pow(xp - xm, 2) + std::pow(yp - ym, 2) + std::pow(zp - zm, 2));
        const amrex::Real s = 0.5 * (a + b + c);
        const amrex::Real asa = 2. * sqrt(s * (s - a) * (s - b) * (s - c));
        // surface area check
        EXPECT_NEAR(sa, asa, tol);
    }
    // Diagonal plane in 3D, wedge of cell
    {
        // Input quantities: normal vector and vof
        const amrex::Real normx = (yp - ym) * (zp - zm);
        const amrex::Real normy = (xp - xm) * (zp - zm);
        const amrex::Real normz = 2.0 * (xp - xm) * (yp - ym);

        // Area of liquid tetrahedron / area of cell
        const amrex::Real VOF =
            ((0.5 * (zp - zm)) * (0.5 * (xp - xm) * (yp - ym))) / 3. /
            ((xp - xm) * (yp - ym) * (zp - zm));

        // Account for normalization and modify for unit cell
        const amrex::Real unitnormx = normx * (xp - xm);
        const amrex::Real unitnormy = normy * (yp - ym);
        const amrex::Real unitnormz = normz * (zp - zm);
        const amrex::Real nmag = unitnormx + unitnormy + unitnormz;

        // Get alpha based on inputs
        amrex::Real alpha = amr_wind::multiphase::volume_intercept(
            unitnormx / nmag, unitnormy / nmag, unitnormz / nmag, VOF);

        // Perform calculations
        amr_wind::multiphase::surfacearea_center(
            normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);

        // Check answers
        constexpr amrex::Real tol = 1e-11;
        // centroid is average of triangular surface nodes
        EXPECT_NEAR(xc, (2. * xm + xp) / 3., tol);
        EXPECT_NEAR(yc, (2. * ym + yp) / 3., tol);
        EXPECT_NEAR(zc, (2.5 * zm + 0.5 * zp) / 3., tol);
        // analytical surface area
        const amrex::Real a =
            sqrt(std::pow((xp - xm), 2) + std::pow(0.5 * (zp - zm), 2));
        const amrex::Real b =
            sqrt(std::pow((yp - ym), 2) + std::pow(0.5 * (zp - zm), 2));
        const amrex::Real c = sqrt(std::pow(xp - xm, 2) + std::pow(yp - ym, 2));
        const amrex::Real s = 0.5 * (a + b + c);
        const amrex::Real asa = sqrt(s * (s - a) * (s - b) * (s - c));
        // surface area check
        EXPECT_NEAR(sa, asa, tol);
    }
}

} // namespace amr_wind_tests

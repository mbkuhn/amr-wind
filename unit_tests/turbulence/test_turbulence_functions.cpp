#include "aw_test_utils/MeshTest.H"
#include "amr-wind/equation_systems/tke/source_terms/KsgsM84Src.H"

namespace amr_wind_tests {

class TurbFuncTest : public MeshTest
{};

TEST_F(TurbFuncTest, test_ceps_local)
{
    amrex::Real out = amr_wind::pde::tke::calc_ceps_local(0.1, 0.4, 0.03);
    constexpr amrex::Real tol = 1e-12;

    EXPECT_NEAR(out, 1.081362007168459, tol);

    out = amr_wind::pde::tke::calc_ceps_local(0.1, 0.0, 0.0);
    EXPECT_TRUE(std::isnan(out));

    out = amr_wind::pde::tke::calc_ceps_local(0.0, 0.4, 0.03);
    EXPECT_EQ(out, 0.0);
}

TEST_F(TurbFuncTest, test_dissip_term)
{
    amrex::Real out = amr_wind::pde::tke::calc_dissip(0.1, 0.4, 0.03);
    constexpr amrex::Real tol = 1e-12;

    EXPECT_NEAR(out, 0.843274042711568, tol);

    out = amr_wind::pde::tke::calc_dissip(0.1, -0.4, 0.03);
    EXPECT_TRUE(std::isnan(out));

    out = amr_wind::pde::tke::calc_dissip(0.1, 0.4, 0.03);
    EXPECT_GT(out, 0.0);
}

} // namespace amr_wind_tests

#include "gtest/gtest.h"
#include "ks_test_utils/MeshTest.H"
#include "src/boundary_conditions/field_boundary_fill/Flather.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {
void initialize_vof(
    kynema_sgf::Field& vof,
    const amrex::Vector<amrex::Geometry>& geom,
    amrex::Real wlev)
{
    for (int lev = 0; lev < vof.repo().num_active_levels(); ++lev) {
        auto& vof_mfab = vof(lev);
        auto vof_arrs = vof_mfab.arrays();
        const auto& zlo = geom[lev].ProbLo(2);
        const auto& dz = geom[lev].CellSize(2);
        amrex::ParallelFor(
            vof_mfab, amrex::IntVect(1),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                const amrex::Real z = zlo + (k + 0.5_rt) * dz;

                if (z + 0.5_rt * dz <= wlev) {
                    vof_arrs[nbx](i, j, k) = 1.0_rt;
                } else if (z - 0.5_rt * dz >= wlev) {
                    vof_arrs[nbx](i, j, k) = 0.0_rt;
                } else {
                    vof_arrs[nbx](i, j, k) = (wlev - (z - 0.5_rt * dz)) / dz;
                }
            });
    }
}
} // namespace

class FlatherBoundaryAverageTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0_rt, 0.0_rt, 0.0_rt}};
            amrex::Vector<amrex::Real> probhi{{8.0_rt, 8.0_rt, 8.0_rt}};
            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
            amrex::Vector<int> periodic{{0, 0, 0}};
            pp.addarr("is_periodic", periodic);
        }
        {
            amrex::ParmParse pp("amr");
            const amrex::Vector<int> ncell{{m_nx, m_nx, m_nx}};
            pp.add("max_level", 1);
            pp.add("max_grid_size", m_nx);
            pp.add("blocking_factor", 2);
            pp.addarr("n_cell", ncell);
        }

        std::stringstream ss;
        ss << "1 // Number of levels" << '\n';
        ss << "1 // Number of boxes at this level" << '\n';
        ss << "0 0 2 4 6 6" << '\n';

        create_mesh_instance<RefineMesh>();
        std::unique_ptr<kynema_sgf::CartBoxRefinement> box_refine(
            new kynema_sgf::CartBoxRefinement(sim()));
        box_refine->read_inputs(mesh(), ss);

        if (mesh<RefineMesh>() != nullptr) {
            mesh<RefineMesh>()->refine_criteria_vec().push_back(
                std::move(box_refine));
        }
    }

    const int m_nx{32};
    const amrex::Real m_wlev{4.0_rt};
};

TEST_F(FlatherBoundaryAverageTest, accumulate_boundary_multilevel)
{
    constexpr amrex::Real u0 = 2.0_rt;
    constexpr amrex::Real v0 = 3.0_rt;
    constexpr amrex::Real tol =
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

    populate_parameters();
    initialize_mesh();

    auto& repo = mesh().field_repo();
    auto& velocity = repo.declare_field("velocity", 3, 1);
    repo.declare_face_normal_field({"u_mac", "v_mac", "w_mac"}, 1, 1, 1);
    auto& vof = repo.declare_field("vof", 1, 1);

    velocity.setVal(u0, 0, 1, 1);
    velocity.setVal(v0, 1, 1, 1);
    velocity.setVal(0.0_rt, 2, 1, 1);
    initialize_vof(vof, mesh().Geom(), m_wlev);

    kynema_sgf::Flather flather(sim());

    kynema_sgf::MultiLevelVector xlo_uavg;
    kynema_sgf::MultiLevelVector xlo_havg;
    kynema_sgf::MultiLevelVector xhi_uavg;
    kynema_sgf::MultiLevelVector xhi_havg;
    kynema_sgf::MultiLevelVector ylo_uavg;
    kynema_sgf::MultiLevelVector ylo_havg;
    kynema_sgf::MultiLevelVector yhi_uavg;
    kynema_sgf::MultiLevelVector yhi_havg;

    xlo_uavg.resize(0, mesh().Geom());
    xlo_havg.resize(0, mesh().Geom());
    xhi_uavg.resize(0, mesh().Geom());
    xhi_havg.resize(0, mesh().Geom());
    ylo_uavg.resize(1, mesh().Geom());
    ylo_havg.resize(1, mesh().Geom());
    yhi_uavg.resize(1, mesh().Geom());
    yhi_havg.resize(1, mesh().Geom());

    const int nlevels = repo.num_active_levels();
    EXPECT_EQ(nlevels, 2);

    for (int lev = 0; lev < nlevels; ++lev) {
        flather.accumulate_boundary(
            lev, 0, true, xlo_uavg, xlo_havg, false,
            kynema_sgf::FieldState::New);
        flather.accumulate_boundary(
            lev, 0, false, xhi_uavg, xhi_havg, false,
            kynema_sgf::FieldState::New);
        flather.accumulate_boundary(
            lev, 1, true, ylo_uavg, ylo_havg, false,
            kynema_sgf::FieldState::New);
        flather.accumulate_boundary(
            lev, 1, false, yhi_uavg, yhi_havg, false,
            kynema_sgf::FieldState::New);

        const auto xhi_idx = xhi_uavg.ncells(lev) - 1;
        const auto yhi_idx = yhi_uavg.ncells(lev) - 1;

        EXPECT_NEAR(xlo_uavg.host_data(lev)[0], u0, tol);
        EXPECT_NEAR(xhi_uavg.host_data(lev)[xhi_idx], u0, tol);
        EXPECT_NEAR(ylo_uavg.host_data(lev)[0], v0, tol);
        EXPECT_NEAR(yhi_uavg.host_data(lev)[yhi_idx], v0, tol);

        EXPECT_NEAR(xlo_havg.host_data(lev)[0], m_wlev, tol);
        EXPECT_NEAR(xhi_havg.host_data(lev)[xhi_idx], m_wlev, tol);
        EXPECT_NEAR(ylo_havg.host_data(lev)[0], m_wlev, tol);
        EXPECT_NEAR(yhi_havg.host_data(lev)[yhi_idx], m_wlev, tol);
    }
}

TEST_F(
    FlatherBoundaryAverageTest, accumulate_boundary_multilevel_boundary_cells)
{
    constexpr amrex::Real u0 = 4.0_rt;
    constexpr amrex::Real v0 = 1.5_rt;
    constexpr amrex::Real tol =
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

    populate_parameters();
    initialize_mesh();

    auto& repo = mesh().field_repo();
    auto& velocity = repo.declare_field("velocity", 3, 1);
    repo.declare_face_normal_field({"u_mac", "v_mac", "w_mac"}, 1, 1, 1);
    auto& vof = repo.declare_field("vof", 1, 1);

    velocity.setVal(u0, 0, 1, 1);
    velocity.setVal(v0, 1, 1, 1);
    velocity.setVal(0.0_rt, 2, 1, 1);
    initialize_vof(vof, mesh().Geom(), m_wlev);

    kynema_sgf::Flather flather(sim());

    kynema_sgf::MultiLevelVector xlo_uavg;
    kynema_sgf::MultiLevelVector xlo_havg;
    kynema_sgf::MultiLevelVector xhi_uavg;
    kynema_sgf::MultiLevelVector xhi_havg;
    kynema_sgf::MultiLevelVector ylo_uavg;
    kynema_sgf::MultiLevelVector ylo_havg;
    kynema_sgf::MultiLevelVector yhi_uavg;
    kynema_sgf::MultiLevelVector yhi_havg;

    xlo_uavg.resize(0, mesh().Geom());
    xlo_havg.resize(0, mesh().Geom());
    xhi_uavg.resize(0, mesh().Geom());
    xhi_havg.resize(0, mesh().Geom());
    ylo_uavg.resize(1, mesh().Geom());
    ylo_havg.resize(1, mesh().Geom());
    yhi_uavg.resize(1, mesh().Geom());
    yhi_havg.resize(1, mesh().Geom());

    const int nlevels = repo.num_active_levels();
    EXPECT_EQ(nlevels, 2);

    for (int lev = 0; lev < nlevels; ++lev) {
        flather.accumulate_boundary(
            lev, 0, true, xlo_uavg, xlo_havg, true,
            kynema_sgf::FieldState::New);
        flather.accumulate_boundary(
            lev, 0, false, xhi_uavg, xhi_havg, true,
            kynema_sgf::FieldState::New);
        flather.accumulate_boundary(
            lev, 1, true, ylo_uavg, ylo_havg, true,
            kynema_sgf::FieldState::New);
        flather.accumulate_boundary(
            lev, 1, false, yhi_uavg, yhi_havg, true,
            kynema_sgf::FieldState::New);

        EXPECT_NEAR(xlo_uavg.host_data(lev)[0], u0, tol);
        EXPECT_NEAR(xhi_uavg.host_data(lev)[0], u0, tol);
        EXPECT_NEAR(ylo_uavg.host_data(lev)[0], v0, tol);
        EXPECT_NEAR(yhi_uavg.host_data(lev)[0], v0, tol);

        EXPECT_NEAR(xlo_havg.host_data(lev)[0], m_wlev, tol);
        EXPECT_NEAR(xhi_havg.host_data(lev)[0], m_wlev, tol);
        EXPECT_NEAR(ylo_havg.host_data(lev)[0], m_wlev, tol);
        EXPECT_NEAR(yhi_havg.host_data(lev)[0], m_wlev, tol);
    }
}

} // namespace kynema_sgf_tests

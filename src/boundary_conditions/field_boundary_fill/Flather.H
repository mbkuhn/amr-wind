#ifndef FLATHER_H
#define FLATHER_H

#include "src/core/Field.H"
#include "src/CFDSim.H"
#include "src/boundary_conditions/field_boundary_fill/FieldBoundary.H"
#include "src/utilities/trig_ops.H"
#include "AMReX_REAL.H"
#include <limits>

using namespace amrex::literals;

namespace kynema_sgf {

/** Flather-style open boundary for velocity
 *
 *  This boundary reads current boundary ghost-cell values and adjacent interior
 *  values, computes a blended target, and writes updated boundary values.
 */
class Flather : public FieldBoundary::Register<Flather>
{
public:
    static std::string identifier() { return "Flather"; }

    explicit Flather(CFDSim& sim);

    void post_init_actions() override;

    void pre_advance_work() override;

    void post_advance_work() override {}

    void set_velocity(
        int lev,
        amrex::Real time,
        const Field& fld,
        amrex::MultiFab& mfab,
        int dcomp = 0,
        int orig_comp = 0) const override;

private:
    void update_target_velocity();

    const CFDSim& m_sim;
    const kynema_sgf::SimTime& m_time;
    const FieldRepo& m_repo;
    const amrex::AmrCore& m_mesh;
    Field& m_velocity;

    amrex::Real m_relaxation{0.5_rt};
    amrex::Real m_target_weight{0.5_rt};

    amrex::Real m_wind_speed{8.0_rt};
    amrex::Real m_wind_direction{270.0_rt};
    amrex::Vector<amrex::Real> m_uvec{8.0_rt, 0.0_rt, 0.0_rt};

    amrex::Real m_start_time{0.0_rt};
    amrex::Real m_stop_time{std::numeric_limits<amrex::Real>::max()};
    amrex::Real m_degrees_per_sec{0.0_rt};
};

} // namespace kynema_sgf

#endif /* FLATHER_H */

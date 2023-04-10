#include "amr-wind/equation_systems/temperature/source_terms/WallTemperatureFluxForcing.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/FieldUtils.H"
#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/wind_energy/ShearStress.H"

#include "AMReX_ParmParse.H"

namespace amr_wind::pde::temperature {

// FIXME: comments out of date
/** Boussinesq buoyancy source term for ABL simulations
 *
 *  Reads in the following parameters from `ABLMeanBoussinesq` namespace:
 *
 *  - `reference_temperature` (Mandatory) temperature (`T0`) in Kelvin
 *  - `thermal_expansion_coeff` Optional, default = `1.0 / T0`
 *  - `gravity` acceleration due to gravity (m/s)
 *  - `read_temperature_profile`
 *  - `tprofile_filename`
 */
WallTemperatureFluxForcing::WallTemperatureFluxForcing(const CFDSim& sim)
    : m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_temperature(sim.repo().get_field("temperature"))
    , m_density(sim.repo().get_field("density"))
    , m_mo(sim.physics_manager().get<amr_wind::ABL>().abl_wall_function().mo())
    , m_wall_temperature_flux_source(
          sim.repo().get_field("wall_temperature_flux_src_term"))
{

    // some parm parse stuff?
    amrex::ParmParse pp("ABL");
    pp.query("wall_shear_stress_type", m_wall_shear_stress_type);
    pp.query("normal_direction", m_direction);
    AMREX_ASSERT((0 <= m_direction) && (m_direction < AMREX_SPACEDIM));
}

WallTemperatureFluxForcing::~WallTemperatureFluxForcing() = default;

template <typename ShearStress>
void WallTemperatureFluxForcing::compute_wall_src(
    const int& idir,
    const int& kLow,
    const int& kHigh,
    const amrex::Real& dV,
    const amrex::Real& weightLow,
    const amrex::Real& weightHigh,
    const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>& dx,
    const amrex::Box& bx,
    const ShearStress& tau,
    const amrex::Array4<amrex::Real>& src_term,
    const amrex::Array4<amrex::Real>& plotField,
    const amrex::Array4<const amrex::Real>& velocityField,
    const amrex::Array4<const amrex::Real>& temperatureField) const
{
    amrex::ParallelFor(amrex::bdryLo(bx, idir),
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        // Get the local velocity at the cell center adjacent
        // to this wall face.
        const amrex::Real uLow = velocityField(i, j, kLow, 0);
        const amrex::Real vLow = velocityField(i, j, kLow, 1);
        const amrex::Real TLow = temperatureField(i, j, kLow);

        const amrex::Real uHigh = velocityField(i, j, kHigh, 0);
        const amrex::Real vHigh = velocityField(i, j, kHigh, 1);
        const amrex::Real THigh = temperatureField(i, j, kHigh);

        const amrex::Real u = weightLow * uLow + weightHigh * uHigh;
        const amrex::Real v = weightLow * vLow + weightHigh * vHigh;
        const amrex::Real S = std::sqrt((u * u) + (v * v));
        const amrex::Real T = weightLow * TLow + weightHigh * THigh;

        // Get local tau_wall based on the local conditions and
        // mean state based on Monin-Obukhov similarity.
        amrex::Real q = tau.calc_theta(S, T);

        // Adding the source term as surface temperature flux times surface
        // area divided by cell volume (division by cell volume is to make
        // this a source per unit volume).
        plotField(i, j, k) = q;
        src_term(i, j, k) += (q * dx[0] * dx[1]) / dV;
    });    
}

void WallTemperatureFluxForcing::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{
    // Overall geometry information
    const auto& geom = m_mesh.Geom(lev);

    // Mesh cell size information
    const auto& dx = m_mesh.Geom(lev).CellSizeArray();

    // Domain size information.
    const auto& domain = geom.Domain();
    amrex::Real dV = 1.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        dV *= dx[dir];
    }

    // Direction
    const int idir = m_direction;

    // Velocity field
    const auto& velocityField =
        m_velocity.state(field_impl::dof_state(fstate))(lev).const_array(mfi);

    // Temperature field
    const auto& temperatureField =
        m_temperature.state(field_impl::dof_state(fstate))(lev).const_array(
            mfi);
    
    // Plot field
    auto plotField = m_wall_temperature_flux_source(lev).array(mfi);

    // Get the desired sampling height.
    const amrex::Real zref = m_mo.zref;

    // Figure out on what grid level the sampling should occur and the
    // interpolation weights.
    const amrex::Real index = (zref / dx[idir]) - 0.5;

    int kLow = int(std::floor(index));
    int kHigh = int(std::ceil(index));
    // if kLow and kHigh point to the same grid cell, separate them by one cell.
    if (kLow == kHigh) {
        kHigh += 1;
    }
    // if kLow lies below the grid box, bump it up to the first grid level.
    if (kLow < 0) {
        kLow = 0;
        kHigh = kLow + 1;
    }
    // if kHigh lies outside the grid box, bump it down to the last grid level.
    else if (kHigh > bx.bigEnd(idir)) {
        kHigh = bx.bigEnd(idir);
        kLow = kHigh - 1;
    }

    const amrex::Real weightLow = amrex::Real(kHigh) - index;
    const amrex::Real weightHigh = index - amrex::Real(kLow);

    if (!(bx.smallEnd(idir) == domain.smallEnd(idir))) return;
    if (idir != 2) return;

    if (m_wall_shear_stress_type == "constant") {
        auto tau = ShearStressConstant(m_mo);
        compute_wall_src(idir, kLow, kHigh, dV, weightLow, weightHigh, dx,
                         bx, tau, src_term, plotField, velocityField, temperatureField);
    } else if (m_wall_shear_stress_type == "default") {
        auto tau = ShearStressDefault(m_mo);
        compute_wall_src(idir, kLow, kHigh, dV, weightLow, weightHigh, dx,
                         bx, tau, src_term, plotField, velocityField, temperatureField);
    } else if (m_wall_shear_stress_type == "local") {
        auto tau = ShearStressLocal(m_mo);
        compute_wall_src(idir, kLow, kHigh, dV, weightLow, weightHigh, dx,
                         bx, tau, src_term, plotField, velocityField, temperatureField);
    } else if (m_wall_shear_stress_type == "schumann") {
        auto tau = ShearStressSchumann(m_mo);
        compute_wall_src(idir, kLow, kHigh, dV, weightLow, weightHigh, dx,
                         bx, tau, src_term, plotField, velocityField, temperatureField);
    } else {
        auto tau = ShearStressMoeng(m_mo);
        compute_wall_src(idir, kLow, kHigh, dV, weightLow, weightHigh, dx,
                         bx, tau, src_term, plotField, velocityField, temperatureField);
    }
}
} // namespace amr_wind::pde::temperature

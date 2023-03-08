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

    //
    const int idir = m_direction;

    const auto& velocityField =
        m_velocity.state(field_impl::dof_state(fstate))(lev).const_array(mfi);

    const auto& temperatureField =
        m_temperature.state(field_impl::dof_state(fstate))(lev).const_array(
            mfi);

    auto plotField = m_wall_temperature_flux_source(lev).array(mfi);

    // FieldState densityState = field_impl::phi_state(fstate);
    // const auto& density =
    // m_density.state(densityState)(lev).const_array(mfi);

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
    // kLow = (kLow < 0) ? 0 : kLow;
    // kHigh = (kLow == kHigh) ? kHigh + 1 : kHigh;
    // kHigh = (kHigh > bx.bigEnd(idir)) ? bx.bigEnd(idir) : kHigh;
    // kLow = (kLow == kHigh) ? kLow - 1 : kLow;

    const amrex::Real weightLow = amrex::Real(kHigh) - index;
    const amrex::Real weightHigh = index - amrex::Real(kLow);

    /*
        std::cout << "index = " << index << std::endl;
        std::cout << "kLow = " << kLow << std::endl;
        std::cout << "kHigh = " << kHigh << std::endl;
        std::cout << "weightLow = " << weightLow << std::endl;
        std::cout << "weightHigh = " << weightHigh << std::endl;
    */

    if (!(bx.smallEnd(idir) == domain.smallEnd(idir))) return;
    if (idir != 2) return;

    amrex::ParallelFor(
        amrex::bdryLo(bx, idir),
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
            /*
                        std::cout << "u = " << uLow << " " << uHigh << " " << u
               << std::endl; std::cout << "v = " << vLow << " " << vHigh << " "
               << v << std::endl; std::cout << "S = " << S << std::endl;
                        std::cout << "T = " << TLow << " " << THigh << " " << T
               << std::endl;
            */

            // Get local tau_wall based on the local conditions and
            // mean state based on Monin-Obukhov similarity.
            amrex::Real q = 0.0;

            if (m_wall_shear_stress_type == "constant") {
                auto tau = ShearStressConstant(m_mo);
                q = tau.calc_theta(S, T);
            } else if (m_wall_shear_stress_type == "default") {
                auto tau = ShearStressDefault(m_mo);
                q = tau.calc_theta(S, T);
            } else if (m_wall_shear_stress_type == "local") {
                auto tau = ShearStressLocal(m_mo);
                q = tau.calc_theta(S, T);
            } else if (m_wall_shear_stress_type == "schumann") {
                auto tau = ShearStressSchumann(m_mo);
                q = tau.calc_theta(S, T);
            } else {
                auto tau = ShearStressMoeng(m_mo);
                q = tau.calc_theta(S, T);
            }

            //          std::cout << "Stheta_mean = " << m_mo.Stheta_mean <<
            //          std::endl;
            /*
                        std::cout << "q = " << q << std::endl;
                        std::cout << "utau = " << m_mo.utau << std::endl;
                        std::cout << "z0 = " << m_mo.z0 << std::endl;
                        std::cout << "z1 = " << m_mo.zref << std::endl;
                        std::cout << "L = " << m_mo.L << std::endl;
                        std::cout << "VLarge = " <<
               std::numeric_limits<amrex::Real>::max() << std::endl; std::cout
               << "phi_m = " << m_mo.phi_m() << std::endl; std::cout << "phi_h =
               " << m_mo.phi_h() << std::endl; std::cout << "psi_m = " <<
               m_mo.psi_m(m_mo.zref/m_mo.obukhov_L) << std::endl; std::cout <<
               "vel_mean = " << m_mo.vel_mean[0] << " "
                                                   << m_mo.vel_mean[1] << " "
                                                   << m_mo.vel_mean[2] <<
               std::endl; std::cout << "vel_current = " << velocityField(i, j,
               k, 0) << " "
                                                      << velocityField(i, j, k,
               1) << " "
                                                      << velocityField(i, j, k,
               2) << std::endl; std::cout << "temp_current = " <<
               temperatureField(i, j, k) << std::endl; std::cout << "temp_mean =
               " << m_mo.theta_mean << std::endl; std::cout << "dx = " << dx[0]
               << " " << dx[1] << " " << dx[2] << std::endl; std::cout <<
               "surf_temp_flux = " << m_mo.surf_temp_flux << std::endl;
                        std::cout << "vMag_mean = " << m_mo.vmag_mean <<
               std::endl; std::cout << "Su_mean = " << m_mo.Su_mean <<
               std::endl; std::cout << "Sv_mean = " << m_mo.Sv_mean <<
               std::endl; std::cout << "Stheta_mean = " << m_mo.Stheta_mean <<
               std::endl; std::cout << "level = " << lev << std::endl; std::cout
               << m_temperature.name() << std::endl; std::cout <<
               field_impl::field_name_with_state(m_temperature.name(),fstate) <<
               std::endl; std::cout << m_velocity.num_states() << std::endl;
                        std::cout << m_velocity.num_time_states() << std::endl;
            */

            // Adding the source term as surface temperature flux times surface
            // area divided by cell volume (division by cell volume is to make
            // this a source per unit volume).
            plotField(i, j, k) = q;

            src_term(i, j, k) += (q * dx[0] * dx[1]) / dV;
        });
}

} // namespace amr_wind::pde::temperature

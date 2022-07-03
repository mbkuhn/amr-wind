#include "amr-wind/ocean_waves/regular_waves/regular_waves_ops.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/utilities/ncutils/nc_interface.H"
#include "amr-wind/utilities/io_utils.H"

#include "amr-wind/fvm/gradient.H"
#include "amr-wind/core/field_ops.H"

// Used for mms
#include "amr-wind/physics/ConvectingTaylorVortex.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace ib {
namespace regular_waves {

void read_inputs(
    RegularWavesBaseData& wdata,
    IBInfo& /*unused*/,
    const ::amr_wind::utils::MultiParser& pp)
{
    pp.query("height", wdata.wave_height);
    pp.query("wavelength", wdata.wave_length);
    pp.query("water_depth", wdata.water_depth);
}

void init_data_structures(RegularWavesBaseData& /*unused*/) {}

void relaxation_zone_free_surface(CFDSim& /*sim*/);
{
    const int nlevels = sim.repo().num_active_levels();
    const auto& levelset = sim.repo().get_field("levelset");
    const auto& vof = sim.repo().get_field("vof");
} // namespace regular_waves

void prepare_netcdf_file(
    const std::string& ncfile,
    const BluffBodyBaseData& meta,
    const IBInfo& info)
{
    amrex::ignore_unused(ncfile, meta, info);
}

void write_netcdf(
    const std::string& ncfile,
    const BluffBodyBaseData& meta,
    const IBInfo& info,
    const amrex::Real time)
{
    amrex::ignore_unused(ncfile, meta, info, time);
}

} // namespace regular_waves
} // namespace ib
} // namespace amr_wind

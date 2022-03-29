#include "amr-wind/ocean_waves/OceanWaves.H"
#include "amr-wind/ocean_waves/WaveModel.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/core/MultiParser.H"

#include <algorithm>

namespace amr_wind {
namespace ocean_waves {

OceanWaves::OceanWaves(CFDSim& sim)
    : m_sim(sim), m_levelset(sim.repo().get_field("levelset"))
{}

OceanWaves::~OceanWaves() = default;

void OceanWaves::pre_init_actions()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::pre_init_actions");
    amrex::ParmParse pp(identifier());

    // Read in the wave type
    std::string type;
    pp.query("type", type);

    AMREX_ALWAYS_ASSERT(!type.empty());
    /*
    auto obj = WaveModel::create(type, m_sim, tname, i);

    const std::string default_prefix = identifier() + "." + type;
    ::amr_wind::utils::MultiParser inp(default_prefix, prefix);

    obj->read_inputs(inp);
    */
}

void OceanWaves::pre_advance_work()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::pre_advance_work");
}

void OceanWaves::post_advance_work()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::post_init_actions");
}

} // namespace ocean_waves
} // namespace amr_wind
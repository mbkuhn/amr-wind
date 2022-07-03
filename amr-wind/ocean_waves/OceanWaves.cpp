#include "amr-wind/ocean_waves/OceanWaves.H"
#include "amr-wind/ocean_waves/OceanWavesModel.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/core/MultiParser.H"

#include <algorithm>

namespace amr_wind {
namespace ocean_waves {

OceanWaves::OceanWaves(CFDSim& sim)
    : m_sim(sim)
    , m_levelset(sim.repo().declare_field("levelset", 1, 1, 1))
    , m_vof(sim.repo().declare_field("vof", 1, 1, 1))
{

    if (!sim.physics_manager().contains("Multiphase")) {
        amrex::Abort("OceanWaves requires Multiphase physics to be active");
    }
    m_levelset.set_default_fillpatch_bc(sim.time());
    m_vof.set_default_fillpatch_bc(sim.time());
}

OceanWaves::~OceanWaves() = default;

void OceanWaves::pre_init_actions()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::pre_init_actions");
    amrex::ParmParse pp(identifier());

    std::string label;
    pp.query("label", label);
    const std::string& tname = label;
    const std::string& prefix = identifier() + "." + tname;
    amrex::ParmParse pp1(prefix);

    std::string type;
    pp.query("type", type);
    pp1.query("type", type);
    AMREX_ALWAYS_ASSERT(!type.empty());

    m_owm = OceanWavesModel::create(type, m_sim, tname, 0);

    const std::string default_prefix = identifier() + "." + type;
    ::amr_wind::utils::MultiParser inp(default_prefix, prefix);

    m_owm->read_inputs(inp);
}

void OceanWaves::post_init_actions()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::post_init_actions");
    m_owm->init_waves();
}

void OceanWaves::post_regrid_actions() {}

void OceanWaves::pre_advance_work()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::pre_advance_work");
    update_relaxation_zones();
}

void OceanWaves::post_advance_work()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::post_init_actions");
}

/** Update ocean waves relaxation zones
 *
 */
void OceanWaves::update_relaxation_zones()
{
    BL_PROFILE("amr-wind::ocean_waves::OceanWaves::update_relaxation_zones");
    m_owm->update_free_surface();
    m_owm->update_velocities();
}

void OceanWaves::prepare_outputs()
{
    const std::string out_dir_prefix = "post_processing/ocean_waves";
    const std::string sname =
        amrex::Concatenate(out_dir_prefix, m_sim.time().time_index());
    if (!amrex::UtilCreateDirectory(sname, 0755)) {
        amrex::CreateDirectoryFailed(sname);
    }

    m_owm->prepare_outputs(sname);
}

} // namespace ocean_waves
} // namespace amr_wind
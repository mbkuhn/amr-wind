#ifndef OCEAN_WAVES_H
#define OCEAN_WAVES_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/FieldRepo.H"

/** \defgroup immersed boundary Immersed boundary module
 *  Immersed boundary class
 *  \ingroup immersed_boundary
 */

namespace amr_wind {

class Field;

namespace ocean_waves {

class OceanWaveModel;

/** ocean waves modeling
 *
 *  \ingroup ocean waves
 *
 *  This class provides an interface to model
 *
 *  \sa WaveModel
 */
class OceanWaves : public Physics::Register<OceanWaves>
{
public:
    static std::string identifier() { return "OceanWaves"; }

    explicit OceanWaves(CFDSim& /*sim*/);

    ~OceanWaves() override;

    void
    initialize_fields(int /*level*/, const amrex::Geometry& /*geom*/) override
    {}

    void pre_init_actions() override;

    void post_init_actions() override {}

    void post_regrid_actions() override {}

    void pre_advance_work() override;

    void post_advance_work() override;

private:
    void relaxation_zones();

    CFDSim& m_sim;

    //! Unique pointer to the ocean waves class
    std::unique_ptr<OceanWaveModel> m_wave_model;

    //! Ocean waves target surface levelset
    Field& m_levelset;
};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* OCEAN_WAVES_H */

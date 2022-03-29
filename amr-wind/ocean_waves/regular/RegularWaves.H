#ifndef REGULAR_WAVES_H
#define REGULAR_WAVES_H

#include "amr-wind/ocean_waves/OceanWaveTypes.H"
#include "amr-wind/core/vs/vector_space.H"

#include <string>

namespace amr_wind {
namespace ocean_waves {

/** Base data representation of a regular waves
 *
 *  \ingroup ocean_waves
 */
struct RegularWaveBaseData
{
    amrex::Real m_wave_length{1.0};
    amrex::Real m_water_depth{1.0};
};

struct RegularWaveType : public OceanWaveType
{};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* REGULAR_WAVES_H 
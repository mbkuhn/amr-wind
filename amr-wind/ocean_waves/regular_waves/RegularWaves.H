#ifndef REGULAR_WAVE_H
#define REGULAR_WAVE_H

#include "amr-wind/ocean_waves/OceanWavesTypes.H"
#include "amr-wind/core/vs/vector_space.H"

#include <string>

namespace amr_wind {
namespace ocean_waves {

/** Base data representation of a regular waves
 *
 *  \ingroup ocean_waves
 */
struct RegularWavesBaseData
{
    amrex::Real wave_height{1.0};

    amrex::Real wave_length{1.0};

    amrex::Real water_depth{1.0};
};

struct RegularWavesType : public OceanWavesType
{};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* REGULAR_WAVES_H 
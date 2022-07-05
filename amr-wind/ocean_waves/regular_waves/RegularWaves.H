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
    amrex::Real zsl{0.0};

    amrex::Real waterdepth{1.0};

    // Wave generation/absorption parameters
    amrex::Real gen_length{4.0};

    amrex::Real beach_length{8.0};

    amrex::Real x_start_beach{22.0};

    bool has_ramp{false};

    amrex::Real ramp_period{2.0};
};

struct RegularWavesType : public OceanWavesType
{};

} // namespace ocean_waves
} // namespace amr_wind

#endif // REGULAR_WAVES_H
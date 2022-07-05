#ifndef LINEARWAVES_H
#define LINEARWAVES_H

#include "amr-wind/ocean_waves/regular_waves/RegularWaves.H"

namespace amr_wind {
namespace ocean_waves {

struct LinearWavesData : public RegularWavesBaseData
{
    amrex::Real wave_height{0.1};
    amrex::Real wave_length{1.0};
};

struct LinearWaves : public RegularWavesType
{
    using InfoType = OceanWavesInfo;
    using MetaType = LinearWavesData;
    using DataType = OceanWavesDataHolder<LinearWaves>;

    static std::string identifier() { return "LinearWaves"; }
};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* LINEARWAVES_H */

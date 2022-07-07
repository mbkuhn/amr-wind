#ifndef STOKES_WAVES_H
#define STOKES_WAVES_H

#include "amr-wind/ocean_waves/regular_waves/RegularWaves.H"

namespace amr_wind {
namespace ocean_waves {

struct StokesWavesData : public RegularWavesBaseData
{
    amrex::Real wave_height{0.1};
    amrex::Real wave_length{1.0};
    int order{2};
};

struct StokesWaves : public RegularWavesType
{
    using InfoType = OceanWavesInfo;
    using MetaType = StokesWavesData;
    using DataType = OceanWavesDataHolder<StokesWaves>;

    static std::string identifier() { return "StokesWaves"; }
};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* STOKESWAVES_H */

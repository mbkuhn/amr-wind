#ifndef HOSWAVES_H
#define HOSWAVES_H

#include "amr-wind/ocean_waves/regular_waves/RegularWaves.H"

namespace amr_wind {
namespace ocean_waves {

struct HOSWavesData : public RegularWavesBaseData
{
    // Prefix for HOS files
    std::string HOS_prefix{"HOSGridData"};
    // Index of first timestep to be used (in filename)
    int HOS_n0{0};
    // Current index
    int HOS_n{0};
    // Index of first x point to be used
    int HOS_i0{0};
    // Index of first y point to be used
    int HOS_j0{0};
    // Total number of HOS points (read from file)
    int HOS_nx{0};
    int HOS_ny{0};
    int HOS_nz{0};
    // Lateral dimensions
    amrex::Real HOS_Lx{0.0};
    amrex::Real HOS_Ly{0.0};
    // Bounds in vertical direction
    amrex::Real HOS_zmin{0.0};
    amrex::Real HOS_zmax{0.0};
    // Time when ow arrays were last changed (last timestep)
    amrex::Real t_last{0.0};
    // Time of current HOS lists
    amrex::Real HOS_t{0.0};
    // Timestep from HOS data
    amrex::Real HOS_dt{0.0};
    // Current data for eta, u, v, and w
    RealList eta;
    RealList u;
    RealList v;
    RealList w;
};

struct HOSWaves : public RegularWavesType
{
    using InfoType = OceanWavesInfo;
    using MetaType = HOSWavesData;
    using DataType = OceanWavesDataHolder<HOSWaves>;

    static std::string identifier() { return "HOSWaves"; }
};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* HOSWAVES_H */

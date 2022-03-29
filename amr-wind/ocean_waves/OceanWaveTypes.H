#ifndef WAVEMODELTYPES_H
#define WAVEMODELTYPES_H

#include <utility>

#include "amr-wind/core/vs/vector_space.H"

#include "AMReX_Gpu.H"
#include "AMReX_RealBox.H"

namespace amr_wind {

class CFDSim;

namespace ocean_waves {

/** Abstract representation of an ocean wave type.
 *
 *  \ingroup immersed boundary
 */
struct OceanWaveType
{};

using RealList = amrex::Vector<amrex::Real>;
using VecList = amrex::Vector<amr_wind::vs::Vector>;
using TensorList = amrex::Vector<amr_wind::vs::Tensor>;
using DeviceVecList = amrex::Gpu::DeviceVector<amr_wind::vs::Vector>;
using DeviceTensorList = amrex::Gpu::DeviceVector<amr_wind::vs::Tensor>;

/** Basic information that is common to all wave models
 *
 *  \ingroup ocean waves
 */
struct OceanWaveInfo
{
    //! Unique string identifier for the ocean waves body (wing, turbine)
    std::string label;

    //! Bounding box used to determine processes where this ocean waves
    amrex::RealBox generation_region;

    amrex::RealBox absorption_refion;

    //! Unique integer identifier for the ocean waves
    int id{-1};

    OceanWaveInfo(std::string label_in, const int id_in)
        : label(std::move(label_in)), id(id_in)
    {}
};

/** Abstract representation of data holder for specific ocean waves types.
 *
 *  \ingroup ocean waves
 *
 */
template <typename OceanWaveTrait>
class OceanWaveDataHolder
{
private:
    //! Instance of the CFD simulation environment
    CFDSim& m_sim;

    //! Basic information about this ocean waves component in relation to
    //! the simulation (usually IBInfo)
    typename OceanWaveTrait::InfoType m_info;

    //! Additional data necessary for a given wavem type
    typename OceanWaveTrait::MetaType m_meta;

public:
    /** Initialize the data structures
     *
     *  \param sim Instance of the CFD simulation environment
     *  \param label Unique string identifier for this immersed boundary
     * component \param id Unique global integer identifier for this immersed
     * boundary component
     */
    OceanWaveDataHolder(CFDSim& sim, const std::string& label, const int id)
        : m_sim(sim), m_info(label, id), m_meta()
    {}

    CFDSim& sim() { return m_sim; }
    const CFDSim& sim() const { return m_sim; }

    typename OceanWaveTrait::InfoType& info() { return m_info; }
    const typename OceanWaveTrait::InfoType& info() const { return m_info; }

    typename OceanWaveTrait::MetaType& meta() { return m_meta; }
    const typename OceanWaveTrait::MetaType& meta() const { return m_meta; }
};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* OCEANTYPES_H */
#ifndef REGULAR_WAVES_OPS_H
#define REGULAR_WAVES_OPS_H

#include "amr-wind/ocean_waves/regular_waves/RegularWaves.H"
#include "amr-wind/ocean_waves/OceanWavesTypes.H"
#include "amr-wind/ocean_waves/OceanWavesOps.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/fvm/gradient.H"

#include "AMReX_MultiFabUtil.H"

namespace amr_wind {
namespace ocean_waves {
namespace regular_waves {

/** Parse common data for modeling regular waves
 */
void read_inputs(
    RegularWavesBaseData& /*wdata*/,
    OceanWavesInfo& /*unused*/,
    const ::amr_wind::utils::MultiParser& /*pp*/);

/** Initialize core data structures when modeling the IB geometry
 */
void init_data_structures(CFDSim&);

/** Set the free_surface height inside the relaxation zones
 */
void apply_relaxation_zones(CFDSim& sim, RegularWavesBaseData&);

void prepare_netcdf_file(
    const std::string& /*ncfile*/,
    const RegularWavesBaseData& /*meta*/,
    const OceanWavesInfo& /*info*/);

void write_netcdf(
    const std::string& /*ncfile*/,
    const RegularWavesBaseData& /*meta*/,
    const OceanWavesInfo& /*info*/,
    const amrex::Real /*time*/);

} // namespace regular_waves

namespace ops {

template <typename WaveTheoryTrait>
struct UseDefaultOp
{
    static constexpr bool update_relaxation_zones = true;
    static constexpr bool apply_relaxation_zones = true;
};

template <typename WaveTheoryTrait>
struct UpdateRelaxZonesOp<
    WaveTheoryTrait,
    typename std::enable_if<
        std::is_base_of<RegularWavesType, WaveTheoryTrait>::value>::type>
{
    void operator()(typename WaveTheoryTrait::DataType& /*unused*/) {}
};

template <typename WaveTheoryTrait>
struct ApplyRelaxZonesOp<
    WaveTheoryTrait,
    typename std::enable_if<
        std::is_base_of<RegularWavesType, WaveTheoryTrait>::value>::type>
{
    void operator()(typename WaveTheoryTrait::DataType& data)
    {
        BL_PROFILE("amr-wind::ocean_waves::OceanWaves::update_velocities");

        auto& wdata = data.meta();
        auto& sim = data.sim();

        regular_waves::apply_relaxation_zones(sim, wdata);
    }
};

template <typename WaveTheoryTrait>
struct ProcessOutputsOp<
    WaveTheoryTrait,
    typename std::enable_if<
        std::is_base_of<RegularWavesType, WaveTheoryTrait>::value>::type>
{
private:
    typename WaveTheoryTrait::DataType& m_data;

    std::string m_out_dir{""};
    std::string m_nc_filename{""};
    int m_out_freq{10};

public:
    explicit ProcessOutputsOp(typename WaveTheoryTrait::DataType& data)
        : m_data(data)
    {}

    void read_io_options(const ::amr_wind::utils::MultiParser& pp)
    {
        pp.query("output_frequency", m_out_freq);
    }

    void prepare_outputs(const std::string& out_dir)
    {
        m_nc_filename = out_dir + "/" + m_data.info().label + ".nc";
        regular_waves::prepare_netcdf_file(
            m_nc_filename, m_data.meta(), m_data.info());
    }

    void write_outputs()
    {
        const auto& time = m_data.sim().time();
        const int tidx = time.time_index();
        if (tidx % m_out_freq != 0) {
            return;
        }

        regular_waves::write_netcdf(
            m_nc_filename, m_data.meta(), m_data.info(), time.new_time());
    }
};

} // namespace ops

} // namespace ocean_waves
} // namespace amr_wind

#endif /* REGULAR_WAVE_OPS_H */

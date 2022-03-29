#ifndef WAVE_MODEL_H
#define WAVE_MODEL_H

#include "amr-wind/core/Factory.H"
#include "amr-wind/ocean_waves/OceanWaveTypes.H"
#include "amr-wind/ocean_waves/OceanWaveOps.H"
#include "amr-wind/core/MultiParser.H"

namespace amr_wind {

class CFDSim;

namespace ocean_waves {

/** Abstract representation of a wave model used for wave generation
 *
 *  \ingroup ocean waves
 *
 * This class is intended to represent a single wave model
 */
class OceanWaveModel
    : public Factory<OceanWaveModel, CFDSim&, const std::string&, const int>
{
public:
    static std::string base_identifier() { return "WaveModel"; }

    ~OceanWaveModel() override = default;

    virtual const std::string& label() const = 0;

    virtual int id() const = 0;

    virtual void read_inputs(const ::amr_wind::utils::MultiParser&) = 0;

    virtual void init_wave() = 0;

    virtual void update_freesurface() = 0;

    virtual void update_velocities() = 0;

    virtual void prepare_outputs(const std::string&) = 0;

    virtual void write_outputs() = 0;
};

/** Concrete implementation of the WaveModel for different wave theories.
 *
 *  \ingroup OceanWaves
 *
 * \tparam WaveTheoryTrait A wave theory trait that defines the behavior of the
 * different actions (initialization, velocity update, free surface update)
 * that are executed by the WaveModel class through the amr_wind::ocean_waves
 * physics driver class.
 *
 */
template <typename WaveTheoryTrait>
class WaveModel : public OceanWaveModel::Register<WaveModel<WaveTheoryTrait>>
{
private:
    //! Instance that holds all data related to a particular immersed boundary
    //! type
    typename WaveTheoryTrait::DataType m_data;

    //! Outputs processor
    ops::ProcessOutputsOp<WaveTheoryTrait> m_out_op;

public:
    static std::string identifier() { return WaveTheoryTrait::identifier(); }

    /**
     *  \param sim CFD simulation environment instance
     *  \param label Unique string identifier for this instance
     *  \param id Unique integer tag for this instance
     */
    WaveModel(CFDSim& sim, const std::string& label, const int id)
        : m_data(sim, label, id), m_out_op(m_data)
    {}

    //! Return the unique identifier (name) for this instance
    const std::string& label() const override { return m_data.info().label; }

    //! Return the unique tag (integer ID) for this instance
    int id() const override { return m_data.info().id; }

    void read_inputs(const ::amr_wind::utils::MultiParser& pp) override
    {
        ops::ReadInputsOp<WaveTheoryTrait>()(m_data, pp);
        m_out_op.read_io_options(pp);
    }

    void update_freesurface() override
    {
        ops::UpdateFSOp<WaveTheoryTrait>()(m_data);
    }

    void update_velocities() override
    {
        ops::UpdateVelOp<WaveTheoryTrait>()(m_data);
    }

    void prepare_outputs(const std::string& out_dir) override
    {
        m_out_op.prepare_outputs(out_dir);
    }

    void write_outputs() override { m_out_op.write_outputs(); }

    void init_wave_model() override
    {
        ops::InitDataOp<WaveTheoryTrait>()(m_data);
    }
};

} // namespace ocean_waves
} // namespace amr_wind

#endif /* IBMODEL_H */

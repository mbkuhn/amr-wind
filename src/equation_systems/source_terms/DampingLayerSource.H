#ifndef DAMPINGLAYERSOURCE_H
#define DAMPINGLAYERSOURCE_H

#include <string>

#include "src/CFDSim.H"
#include "src/core/FieldRepo.H"
#include "src/equation_systems/density/DensitySource.H"
#include "src/equation_systems/sdr/SDRSource.H"
#include "src/equation_systems/SourceTerm.H"
#include "src/equation_systems/icns/MomentumSource.H"
#include "src/equation_systems/temperature/TemperatureSource.H"
#include "src/equation_systems/tke/TKESource.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde {

template <typename T>
struct DampingLayerSourceTraits
{
    static constexpr bool is_specialized = false;
    static constexpr const char* field_name = "";
};

template <>
struct DampingLayerSourceTraits<MomentumSource>
{
    static constexpr bool is_specialized = true;
    static constexpr const char* field_name = "velocity";
};

template <>
struct DampingLayerSourceTraits<TemperatureSource>
{
    static constexpr bool is_specialized = true;
    static constexpr const char* field_name = "temperature";
};

template <>
struct DampingLayerSourceTraits<DensitySource>
{
    static constexpr bool is_specialized = true;
    static constexpr const char* field_name = "density";
};

template <>
struct DampingLayerSourceTraits<TKESource>
{
    static constexpr bool is_specialized = true;
    static constexpr const char* field_name = "tke";
};

template <>
struct DampingLayerSourceTraits<SDRSource>
{
    static constexpr bool is_specialized = true;
    static constexpr const char* field_name = "sdr";
};

template <>
struct DampingLayerSourceTraits<SourceTerm>
{
    static constexpr bool is_specialized = true;
    static constexpr const char* field_name = "";
};

template <typename T>
class DampingLayerSource : public T::template Register<DampingLayerSource<T>>
{
public:
    static_assert(
        DampingLayerSourceTraits<T>::is_specialized,
        "DampingLayerSourceTraits<T> specialization missing for this source "
        "type. Add a specialization with field_name.");

    static std::string identifier() { return "DampingLayerSource"; }

    static constexpr const char* damped_field_name()
    {
        return DampingLayerSourceTraits<T>::field_name;
    }

    explicit DampingLayerSource(const CFDSim& sim)
        : m_repo(sim.repo())
        , m_damped_field_name(damped_field_name())
        , m_damped_field(sim.repo().get_field(damped_field_name()))
    {

        // Insert gets and queries about target field profile, i.e., what to
        // damp towards
        amrex::ParmParse pp(identifier() + "." + m_damped_field_name);
    }

    ~DampingLayerSource() override = default;

    void operator()(
        int lev, FieldState fstate, amrex::MultiFab& src_term) const override
    {

        /*const auto& dx = geom.CellSizeArray();
        const auto& prob_lo = geom.ProbLoArray();
        const auto& prob_hi = geom.ProbHiArray();*/

        Field* damping_layer_ptr{nullptr};

        auto const& src_arrs = src_term.arrays();
        auto const& damped_field_arrs =
            m_damped_field.state(field_impl::dof_state(fstate))(lev)
                .const_arrays();

        for (int bc_idx = 0; bc_idx < 6; ++bc_idx) {
            const auto damping_coefficient_field_name =
                "damping_layer_" + m_damped_field_name + "_" +
                m_bc_names[bc_idx];
            if (m_repo.field_exists(damping_coefficient_field_name)) {
                // Form damping layer field name
                damping_layer_ptr =
                    &m_repo.get_field(damping_coefficient_field_name);
                auto& damping_mfab = (*damping_layer_ptr)(lev);
                auto const& damping_arrs = damping_mfab.const_arrays();

                amrex::ParallelFor(
                    damping_mfab, amrex::IntVect(0), m_damped_field.num_comp(),
                    [=] AMREX_GPU_DEVICE(
                        int nbx, int i, int j, int k, int n) noexcept {
                        // Modify src term based on current value of damped
                        // field, damping coefficient field, target value

                        // Placeholder for now!!!!
                        const auto target_value = 0.0_rt;

                        src_arrs[nbx](i, j, k, n) +=
                            damping_arrs[nbx](i, j, k) *
                            (target_value - damped_field_arrs[nbx](i, j, k, n));
                    });
            }
        }
    }

private:
    FieldRepo& m_repo;
    std::string m_damped_field_name;
    Field& m_damped_field;
    const amrex::Vector<std::string> m_bc_names = {"xlo", "xhi", "ylo",
                                                   "yhi", "zlo", "zhi"};
};

} // namespace kynema_sgf::pde

#endif /* DAMPINGLAYERSOURCE_H */

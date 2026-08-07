#ifndef DAMPINGLAYERSOURCE_H
#define DAMPINGLAYERSOURCE_H

#include <string>

#include "src/CFDSim.H"
#include "src/core/FieldRepo.H"
#include "src/utilities/linear_interpolation.H"
#include "src/equation_systems/density/DensitySource.H"
#include "src/equation_systems/sdr/SDRSource.H"
#include "src/equation_systems/SourceTerm.H"
#include "src/equation_systems/icns/MomentumSource.H"
#include "src/equation_systems/temperature/TemperatureSource.H"
#include "src/equation_systems/tke/TKESource.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

enum class TargetType : std::uint8_t {
    Constant = 0,
    Profile = 1,
    Function = 2,
    Field = 3
};

TargetType string_to_target_type(const std::string& target_type_str)
{
    std::string target_type_lower = amrex::toLower(target_type_str);
    if (target_type_lower == "constant") {
        return TargetType::Constant;
    } else if (target_type_lower == "profile") {
        return TargetType::Profile;
    } else if (target_type_lower == "function") {
        return TargetType::Function;
    } else if (target_type_lower == "field") {
        return TargetType::Field;
    } else {
        amrex::Abort(
            "Invalid target type: " + target_type_str +
            ". Must be one of: constant, profile, function, field.");
    }
    // Unreachable, but avoids compiler warning
    return TargetType::Constant;
}

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
        : m_mesh(sim.mesh())
        , m_repo(sim.repo())
        , m_time(sim.time())
        , m_damped_field_name(damped_field_name())
        , m_damped_field(sim.repo().get_field(damped_field_name()))
    {

        // Insert gets and queries about target field profile
        amrex::ParmParse pp(identifier() + "." + m_damped_field_name);
        std::string target_type_string;
        std::string target_function_string{"0.0"};
        pp.get("target_type", target_type_string);
        m_target_type = string_to_target_type(target_type_string);
        if (m_target_type == TargetType::Constant) {
            pp.getarr("target_value", m_target_value);
        } else if (m_target_type == TargetType::Profile) {
            pp.getarr("target_profile_values", m_target_profile_values);
            pp.getarr("target_profile_heights", m_target_profile_heights);
        } else if (m_target_type == TargetType::Function) {
            pp.get("target_function", target_function_string);
        } else if (m_target_type == TargetType::Field) {
            pp.get("target_field", m_target_field_name);
        } else {
            amrex::Abort(
                "DampingLayerSource: unknown target_type: " +
                target_type_string);
        }

        pp.queryarr("damped_components", m_damped_components);

        if (m_target_type == TargetType::Constant) {
            if (!m_damped_components.empty() &&
                m_target_value.size() != m_damped_components.size()) {
                amrex::Abort(
                    "DampingLayerSource: target_value must have the same "
                    "size as the number of damped components.");
            } else if (m_target_value.size() != m_damped_field.num_comp()) {
                amrex::Abort(
                    "DampingLayerSource: target_value must have the same "
                    "size as the number of components in the damped field.");
            }
            const auto num_comp_val = m_target_value.size();
            m_dev_target_value.resize(num_comp_val);
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, m_target_value.begin(),
                m_target_value.end(), m_dev_target_value.begin());
        }
        if (m_target_type == TargetType::Profile) {
            // What about profiles for multiple-component fields?
            if (m_target_profile_values.size() !=
                m_target_profile_heights.size()) {
                amrex::Abort(
                    "DampingLayerSource: target_profile_values and "
                    "target_profile_heights must have the same size.");
            }
            const auto num_prof_val = m_target_profile_values.size();
            m_dev_target_profile_values.resize(num_prof_val);
            m_dev_target_profile_heights.resize(num_prof_val);

            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, m_target_profile_heights.begin(),
                m_target_profile_heights.end(),
                m_dev_target_profile_heights.begin());
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, m_target_profile_values.begin(),
                m_target_profile_values.end(),
                m_dev_target_profile_values.begin());
        }
        if (m_target_type == TargetType::Function) {
            m_function_parser.define(target_function_string);
            m_function_parser.registerVariables({"t", "x", "y", "z", "n"});
        }
    }

    ~DampingLayerSource() override = default;

    void operator()(
        int lev, FieldState fstate, amrex::MultiFab& src_term) const override
    {

        /*const auto& dx = geom.CellSizeArray();
        const auto& prob_lo = geom.ProbLoArray();
        const auto& prob_hi = geom.ProbHiArray();*/

        const auto target_type = m_target_type;

        Field* damping_layer_ptr{nullptr};

        auto const& src_arrs = src_term.arrays();
        auto const& damped_field_arrs =
            m_damped_field.state(field_impl::dof_state(fstate))(lev)
                .const_arrays();

        const auto& geom = m_mesh.Geom(lev);
        const auto& problo = geom.ProbLoArray();
        const auto& dx = geom.CellSizeArray();
        const auto time = m_time.current_time();

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

                // Get constant target value if being used
                const amrex::Real* const_target_value =
                    (target_type == TargetType::Constant)
                        ? m_dev_target_value.data()
                        : nullptr;

                // Get target profile values, positions if being used
                const bool use_profile = (target_type == TargetType::Profile);
                const amrex::Real* prof_z =
                    use_profile ? m_dev_target_profile_heights.data() : nullptr;
                const amrex::Real* prof_z_end =
                    use_profile ? m_dev_target_profile_heights.end() : nullptr;
                const amrex::Real* prof_v =
                    use_profile ? m_dev_target_profile_values.data() : nullptr;

                // Set up the user-defined function, whether used or not
                auto udf_func = m_function_parser.compile<AMREX_SPACEDIM + 2>();

                // Get target field arrays if being used
                Field* target_field_ptr{nullptr};
                if (target_type == TargetType::Field) {
                    target_field_ptr = &m_repo.get_field(m_target_field_name);
                }
                auto const& target_field_arrs =
                    (target_type == TargetType::Field)
                        ? (*target_field_ptr)(lev).const_arrays()
                        : amrex::MultiArray4<const amrex::Real>();

                amrex::ParallelFor(
                    damping_mfab, amrex::IntVect(0), m_damped_field.num_comp(),
                    [=] AMREX_GPU_DEVICE(
                        int nbx, int i, int j, int k, int n) noexcept {
                        const amrex::Real x =
                            problo[0] + ((i + 0.5_rt) * dx[0]);
                        const amrex::Real y =
                            problo[1] + ((j + 0.5_rt) * dx[1]);
                        const amrex::Real z =
                            problo[2] + ((k + 0.5_rt) * dx[2]);

                        // Modify src term based on current value of damped
                        // field, damping coefficient field, target value
                        auto target_value = 0.0_rt;
                        if (target_type == TargetType::Constant) {
                            // Will need to be smarter about the component
                            // selection!!!!
                            target_value = const_target_value[n];
                        } else if (target_type == TargetType::Profile) {
                            const auto idx =
                                interp::bisection_search(prof_z, prof_z_end, z);

                            // Interpolate the target profile values based on
                            // the cell center height
                            target_value =
                                interp::linear_impl(prof_z, prof_v, z, idx);
                        } else if (target_type == TargetType::Function) {
                            // Evaluate the user-defined function
                            target_value = udf_func(
                                time, x, y, z, static_cast<amrex::Real>(n));
                        } else if (target_type == TargetType::Field) {
                            target_value = target_field_arrs[nbx](i, j, k, n);
                        }

                        bool damp_this_component = true;
                        // Add logic to select component to damp

                        if (damp_this_component) {
                            src_arrs[nbx](i, j, k, n) +=
                                damping_arrs[nbx](i, j, k) *
                                (target_value -
                                 damped_field_arrs[nbx](i, j, k, n));
                        }
                    });
            }
        }
    }

private:
    const amrex::AmrCore& m_mesh;
    const FieldRepo& m_repo;
    const SimTime& m_time;
    std::string m_damped_field_name;
    Field& m_damped_field;
    TargetType m_target_type;
    amrex::Vector<amrex::Real> m_target_value;
    amrex::Vector<amrex::Real> m_target_profile_values;
    amrex::Vector<amrex::Real> m_target_profile_heights;
    std::string m_target_field_name;
    const amrex::Vector<std::string> m_bc_names = {"xlo", "xhi", "ylo",
                                                   "yhi", "zlo", "zhi"};
    amrex::Vector<int> m_damped_components;

    // Device copies of the above arrays
    amrex::Gpu::DeviceVector<amrex::Real> m_dev_target_value;
    amrex::Gpu::DeviceVector<amrex::Real> m_dev_target_profile_values;
    amrex::Gpu::DeviceVector<amrex::Real> m_dev_target_profile_heights;

    // Parser for evaluating user-defined functions on device
    amrex::Parser m_function_parser;
};

} // namespace kynema_sgf::pde

#endif /* DAMPINGLAYERSOURCE_H */

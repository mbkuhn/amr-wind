#include <utility>

#include "src/boundary_conditions/field_boundary_fill/FillMPL.H"

namespace kynema_sgf {

FillMPL::FillMPL(
    Field& field,
    const amrex::AmrCore& mesh,
    const SimTime& time,
    const ModulatedPowerLaw& abl_mpl)
    : FieldFillPatchOps<FieldBCDirichlet>(
          field, mesh, time, FieldInterpolator::CellConsLinear)
    , m_abl_mpl(abl_mpl)
{}

FillMPL::~FillMPL() = default;

void FillMPL::fillpatch(
    const int lev,
    const amrex::Real time,
    amrex::MultiFab& mfab,
    const amrex::IntVect& nghost,
    const FieldState fstate)
{
    if (m_field.base_name() == "velocity") {
        m_abl_mpl.set_velocity(lev, time, m_field, mfab);
    } else if (m_field.base_name() == "temperature") {
        m_abl_mpl.set_temperature(lev, time, m_field, mfab);
    }
}

void FillMPL::fillpatch_from_coarse(
    const int lev,
    const amrex::Real time,
    amrex::MultiFab& mfab,
    const amrex::IntVect& nghost,
    const FieldState fstate)
{
    if (m_field.base_name() == "velocity") {
        m_abl_mpl.set_velocity(lev, time, m_field, mfab);
    } else if (m_field.base_name() == "temperature") {
        m_abl_mpl.set_temperature(lev, time, m_field, mfab);
    }
}

void FillMPL::fillphysbc(
    const int lev,
    const amrex::Real time,
    amrex::MultiFab& mfab,
    const amrex::IntVect& nghost,
    const FieldState fstate)
{
    if (m_field.base_name() == "velocity") {
        m_abl_mpl.set_velocity(lev, time, m_field, mfab);
    } else if (m_field.base_name() == "temperature") {
        m_abl_mpl.set_temperature(lev, time, m_field, mfab);
    }
}

void FillMPL::fillpatch_sibling_fields(
    const int lev,
    const amrex::Real time,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& mfabs,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& ffabs,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& cfabs,
    const amrex::IntVect& nghost,
    const amrex::Vector<amrex::BCRec>& bcrec,
    const amrex::Vector<amrex::BCRec>& /* unused */,
    const FieldState fstate)
{
    if (m_field.base_name() == "velocity") {
        for (int i = 0; std::cmp_less(i, mfabs.size()); i++) {
            m_abl_mpl.set_velocity(lev, time, m_field, *mfabs[i], 0, i);
        }
    }
}

} // namespace kynema_sgf

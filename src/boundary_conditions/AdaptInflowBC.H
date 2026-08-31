#ifndef ADAPTINFLOWBC_H
#define ADAPTINFLOWBC_H

#include "src/core/FieldBCOps.H"
#include "src/core/FieldRepo.H"
#include "AMReX_Orientation.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf {

/** Custom Neumann fills for the adapt-inflow BC
 *  \ingroup field_bc
 *
 *  Extrapolates outflow boundary cells for adapt_inflow BC.
 *  Passive and inflow cells retain their prescribed or enforcement-set values.
 */
class AdaptInflowBC : public FieldBCIface
{
public:
    AdaptInflowBC(Field& field, amrex::Orientation ori);

    void operator()(Field& field, FieldState /*rho_state*/) override;

private:
    Field& m_field;

    amrex::Orientation m_ori;
};

} // namespace kynema_sgf

#endif /* ADAPTINFLOWBC_H */

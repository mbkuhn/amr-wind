#include "src/boundary_conditions/AdaptInflowBC.H"

namespace kynema_sgf {

AdaptInflowBC::AdaptInflowBC(Field& field, amrex::Orientation ori)
    : m_field(field), m_ori(ori)
{}

void AdaptInflowBC::operator()(Field& /*field*/, const FieldState /*rho_state*/)
{
    const auto& repo = m_field.repo();
    const auto* volume_frac =
        repo.field_exists("vof") ? &repo.get_field("vof") : nullptr;
    const int ncomp = m_field.num_comp();
    const int idim = m_ori.coordDir();
    const auto islow = m_ori.isLow();
    const auto ishigh = m_ori.isHigh();
    const int nlevels = m_field.repo().num_active_levels();
    const amrex::IntVect iv_dir = {
        static_cast<int>(idim == 0), static_cast<int>(idim == 1),
        static_cast<int>(idim == 2)};

    const amrex::Real small_val = 1.0e-12_rt;
    bool use_vof = (volume_frac != nullptr);

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& domain = repo.mesh().Geom(lev).Domain();

        amrex::MFItInfo mfi_info{};
        if (amrex::Gpu::notInLaunchRegion()) {
            mfi_info.SetDynamic(true);
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(m_field(lev), mfi_info); mfi.isValid(); ++mfi) {
            auto bx = mfi.validbox();
            bx.grow(
                {static_cast<int>(idim != 0), static_cast<int>(idim != 1),
                 static_cast<int>(idim != 2)});
            const auto& bc_a = m_field(lev).array(mfi);
            const auto vof = volume_frac ? (*volume_frac)(lev).const_array(mfi)
                                         : amrex::Array4<amrex::Real const>();

            // For outflow cells (velocity at ghost/boundary cell points
            // outward), extrapolate by copying interior values to the boundary
            // ghost cell.
            if (islow && (bx.smallEnd(idim) == domain.smallEnd(idim))) {
                amrex::ParallelFor(
                    amrex::bdryLo(bx, idim),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        const amrex::IntVect iv = {i, j, k};
                        const amrex::IntVect ivm = iv - iv_dir;
                        if (use_vof && vof(ivm) < small_val &&
                            vof(iv) < small_val) {
                            for (int n = 0; n < ncomp; n++) {
                                bc_a(ivm, n) = bc_a(iv, n);
                            }
                        }
                    });
            }

            if (ishigh && (bx.bigEnd(idim) == domain.bigEnd(idim))) {
                amrex::ParallelFor(
                    amrex::bdryHi(bx, idim),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        const amrex::IntVect iv = {i, j, k};
                        const amrex::IntVect ivm = iv - iv_dir;
                        if (use_vof && vof(ivm) < small_val &&
                            vof(iv) < small_val) {
                            for (int n = 0; n < ncomp; n++) {
                                bc_a(iv, n) = bc_a(ivm, n);
                            }
                        }
                    });
            }
        }
    }
}

} // namespace kynema_sgf

#ifndef CURVATURE_H_
#define CURVATURE_H_

#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "amr-wind/../submods/eigen/Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace amr_wind::multiphase {
// Perform local paraboloid fit of interface surface in pointwise sense
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real paraboloid_fit(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real const> const& volfrac,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
    // Structure for linear solve
    MatrixXd A(125, 6);
    VectorXd b(125);
    VectorXd sol(6);
    // Effective mesh size
    const amrex::Real meshsize = std::pow(dx[0] * dx[1] * dx[2], 1.0 / 3.0);
    std::cout << meshsize << std::endl;

    // Calculate local interface plane
    amrex::Real mx = 1.0;
    amrex::Real my = 1.0;
    amrex::Real mz = 1.0;
    amrex::Real alpha = 0.0;
    amr_wind::multiphase::fit_plane(i, j, k, volfrac, mx, my, mz, alpha);
    // Address non-cubic cells
    amrex::Real normx = mx / dx[0];
    amrex::Real normy = my / dx[1];
    amrex::Real normz = mz / dx[2];
    // Normalize by magnitude
    amrex::Real nmag = sqrt(normx * normx + normy * normy + normz * normz);
    normx /= nmag;
    normy /= nmag;
    normz /= nmag;

    // Store this plane as a reference
    const amrex::Real normxref = normx;
    const amrex::Real normyref = normy;
    const amrex::Real normzref = normz;
    std::cout << normxref << " " << normyref << " " << normzref << std::endl;

    // Calculate local surface centroid
    amrex::Real xc = 0.0;
    amrex::Real yc = 0.0;
    amrex::Real zc = 0.0;
    amrex::Real sa = 0.0;
    amrex::Real xm = problo[0] + dx[0] * i;
    amrex::Real xp = xm + dx[0];
    amrex::Real ym = problo[1] + dx[1] * j;
    amrex::Real yp = ym + dx[1];
    amrex::Real zm = problo[2] + dx[2] * k;
    amrex::Real zp = zm + dx[2];
    amr_wind::multiphase::surfacearea_center(
        normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc, yc, zc);
    // Store this place as a reference
    const amrex::Real xref = xc;
    const amrex::Real yref = yc;
    const amrex::Real zref = zc;

    // Create local basis from polygon normal
    amrex::Real tr0, tr1, tr2;
    if (amrex::Math::abs(normx) >
        amrex::max(amrex::Math::abs(normy), amrex::Math::abs(normz))) {
        nmag = sqrt(normy * normy + normx * normx);
        tr0 = normy / nmag;
        tr1 = -normx / nmag;
        tr2 = 0.0;
    } else if (
        amrex::Math::abs(normy) >
        amrex::max(amrex::Math::abs(normx), amrex::Math::abs(normz))) {
        nmag = sqrt(normy * normy + normz * normz);
        tr0 = 0.0;
        tr1 = normz / nmag;
        tr2 = -normy / nmag;
    } else {
        nmag = sqrt(normz * normz + normx * normx);
        tr0 = -normz / nmag;
        tr1 = 0.0;
        tr2 = normx / nmag;
    }
    // Cross product: norm x tr
    const amrex::Real sr0 = normy * tr2 - normz * tr1;
    const amrex::Real sr1 = normz * tr0 - normx * tr2;
    const amrex::Real sr2 = normx * tr1 - normy * tr0;

    // Collect data for linear system
    int ndata = 0;
    for (int ii = i - 2; ii <= i + 2; ++ii) {
        for (int jj = j - 2; jj <= j + 2; ++jj) {
            for (int kk = k - 2; kk <= k + 2; ++kk) {

                // Skip cells with no planes
                if ((volfrac(ii, jj, kk) > 1.0 - 1e-12) ||
                    (volfrac(ii, jj, kk) < 1e-12)) {
                    continue;
                }

                // std::cout << ii << " " << jj << " " << kk << " " <<
                // volfrac(ii,jj,kk) << std::endl;

                // Get local plane normal
                amr_wind::multiphase::fit_plane(
                    ii, jj, kk, volfrac, mx, my, mz, alpha);

                // Address non-cubic cells
                normx = mx / dx[0];
                normy = my / dx[1];
                normz = mz / dx[2];
                // Normalize by magnitude
                nmag = sqrt(normx * normx + normy * normy + normz * normz);
                normx /= nmag;
                normy /= nmag;
                normz /= nmag;

                // Get local plane centroid and normalized surface area
                xm = problo[0] + dx[0] * ii;
                xp = xm + dx[0];
                ym = problo[1] + dx[1] * jj;
                yp = ym + dx[1];
                zm = problo[2] + dx[2] * kk;
                zp = zm + dx[2];
                amr_wind::multiphase::surfacearea_center(
                    normx, normy, normz, alpha, xm, xp, ym, yp, zm, zp, sa, xc,
                    yc, zc);
                sa /= meshsize * meshsize;

                // Transform centroid to local coord system
                const amrex::Real plocx =
                    ((xc - xref) * normxref + (yc - yref) * normyref +
                     (zc - zref) * normzref) /
                    meshsize;
                const amrex::Real plocy =
                    ((xc - xref) * tr0 + (yc - yref) * tr1 +
                     (zc - zref) * tr2) /
                    meshsize;
                const amrex::Real plocz =
                    ((xc - xref) * sr0 + (yc - yref) * sr1 +
                     (zc - zref) * sr2) /
                    meshsize;

                // Weighted distance from reference
                const amrex::Real gd =
                    sqrt(plocx * plocx + plocy * plocy + plocz * plocz);
                const amrex::Real gh = 2.5;
                const amrex::Real wgauss =
                    (gd >= gh)
                        ? 0.0
                        : (1.0 + 4.0 * gd / gh) * std::pow(1.0 - gd / gh, 4);

                const amrex::Real ww =
                    sa *
                    (amrex::max(
                        0.0, normx * normxref + normy * normyref +
                                 normz * normzref)) *
                    wgauss;
                // Add to the least-squares problem if present
                if (ww > 1e-12) {
                    // Increment counter
                    ++ndata;
                    // Store matrix and vector
                    A(ndata, 0) = sqrt(ww);
                    A(ndata, 1) = sqrt(ww) * yc;
                    A(ndata, 2) = sqrt(ww) * zc;
                    A(ndata, 3) = sqrt(ww) * 0.5 * yc * yc;
                    A(ndata, 4) = sqrt(ww) * 0.5 * zc * zc;
                    A(ndata, 5) = sqrt(ww) * yc * zc;
                    b(ndata) = sqrt(ww) * xc;
                    /*std::cout << ndata << " " << A(ndata,0) << " " <<
                       A(ndata,1) << " " << A(ndata,2) << " " << A(ndata,3) << "
                       " << A(ndata,4) << " " << A(ndata,5) << " " << b(ndata)
                       << std::endl;*/
                } else {
                    /*std::cout << ii << " " << jj << " " << kk << " " << wgauss
                              << " " << plocx << " " << plocy << " " << plocz
                              << std::endl;*/
                }
            }
        }
    }
    // Zero other values in linear system
    for (int n = ndata + 1; n < 125; ++n) {
        A(n, 0) = 0.0;
        A(n, 1) = 0.0;
        A(n, 2) = 0.0;
        A(n, 3) = 0.0;
        A(n, 4) = 0.0;
        A(n, 5) = 0.0;
        b(n) = 0.0;
    }
    // Solve linear system
    //sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    sol = A.colPivHouseholderQr().solve(b);

    // Compute curvature
    const amrex::Real dF_dt = sol(1);
    const amrex::Real ddF_dtdt = sol(3);
    const amrex::Real dF_ds = sol(2);
    const amrex::Real ddF_dsds = sol(4);
    const amrex::Real ddF_dtds = sol(5);
    const amrex::Real mycurv =
        -((1.0 + dF_dt * dF_dt) * ddF_dsds - 2.0 * dF_dt * dF_ds * ddF_dtds +
          (1.0 + dF_ds * dF_ds) * ddF_dtdt) /
        std::pow(1.0 + dF_dt * dF_dt + dF_ds * dF_ds, 1.5);
    return (mycurv); // / meshsize);
}
} // namespace amr_wind::multiphase

#endif // CURVATURE.H
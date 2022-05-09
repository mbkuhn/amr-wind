#include "amr-wind/convection/incflo_godunov_plm.H"
#include "amr-wind/convection/incflo_godunov_ppm.H"
#include "amr-wind/convection/incflo_godunov_ppm_nolim.H"
#include "amr-wind/convection/incflo_godunov_weno.H"
#include "amr-wind/convection/incflo_godunov_minmod.H"
#include "amr-wind/convection/Godunov.H"
#include <AMReX_Geometry.H>

using namespace amrex;

void godunov::compute_fluxes(
    int lev,
    Box const& bx,
    int ncomp,
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz,
    Array4<Real const> const& q,
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac,
    Array4<Real const> const& fq,
    BCRec const* pbc,
    int const* iconserv,
    Real* p,
    Vector<amrex::Geometry> geom,
    Real dt,
    godunov::scheme godunov_scheme,
    Array4<int const> const& flag)
{

    BL_PROFILE("amr-wind::godunov::compute_fluxes");
    Box const& xbx = amrex::surroundingNodes(bx, 0);
    Box const& ybx = amrex::surroundingNodes(bx, 1);
    Box const& zbx = amrex::surroundingNodes(bx, 2);
    Box const& bxg1 = amrex::grow(bx, 1);
    Box xebox = Box(xbx).grow(1, 1).grow(2, 1);
    Box yebox = Box(ybx).grow(0, 1).grow(2, 1);
    Box zebox = Box(zbx).grow(0, 1).grow(1, 1);

    const Real dx = geom[lev].CellSize(0);
    const Real dy = geom[lev].CellSize(1);
    const Real dz = geom[lev].CellSize(2);
    Real l_dt = dt;
    Real dtdx = l_dt / dx;
    Real dtdy = l_dt / dy;
    Real dtdz = l_dt / dz;

    Box const& domain = geom[lev].Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);

    Array4<Real> Imx = makeArray4(p, bxg1, ncomp);
    p += Imx.size();
    Array4<Real> Ipx = makeArray4(p, bxg1, ncomp);
    p += Ipx.size();
    Array4<Real> Imy = makeArray4(p, bxg1, ncomp);
    p += Imy.size();
    Array4<Real> Ipy = makeArray4(p, bxg1, ncomp);
    p += Ipy.size();
    Array4<Real> Imz = makeArray4(p, bxg1, ncomp);
    p += Imz.size();
    Array4<Real> Ipz = makeArray4(p, bxg1, ncomp);
    p += Ipz.size();
    Array4<Real> xlo = makeArray4(p, xebox, ncomp);
    p += xlo.size();
    Array4<Real> xhi = makeArray4(p, xebox, ncomp);
    p += xhi.size();
    Array4<Real> ylo = makeArray4(p, yebox, ncomp);
    p += ylo.size();
    Array4<Real> yhi = makeArray4(p, yebox, ncomp);
    p += yhi.size();
    Array4<Real> zlo = makeArray4(p, zebox, ncomp);
    p += zlo.size();
    Array4<Real> zhi = makeArray4(p, zebox, ncomp);
    p += zhi.size();
    Array4<Real> xyzlo = makeArray4(p, bxg1, ncomp);
    p += xyzlo.size();
    Array4<Real> xyzhi = makeArray4(p, bxg1, ncomp);

    // Use PPM to generate Im and Ip */
    switch (godunov_scheme) {
    case godunov::scheme::PPM: {
        amrex::ParallelFor(
            bxg1, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_ppm_fpu_x(
                    i, j, k, n, l_dt, dx, Imx(i, j, k, n), Ipx(i, j, k, n), q,
                    umac, pbc[n], dlo.x, dhi.x);
                Godunov_ppm_fpu_y(
                    i, j, k, n, l_dt, dy, Imy(i, j, k, n), Ipy(i, j, k, n), q,
                    vmac, pbc[n], dlo.y, dhi.y);
                Godunov_ppm_fpu_z(
                    i, j, k, n, l_dt, dz, Imz(i, j, k, n), Ipz(i, j, k, n), q,
                    wmac, pbc[n], dlo.z, dhi.z);
            });
        break;
    }
    case godunov::scheme::PPM_NOLIM: {
        amrex::ParallelFor(
            bxg1, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_ppm_fpu_x_nolim(
                    i, j, k, n, l_dt, dx, Imx(i, j, k, n), Ipx(i, j, k, n), q,
                    umac, pbc[n], dlo.x, dhi.x);
                Godunov_ppm_fpu_y_nolim(
                    i, j, k, n, l_dt, dy, Imy(i, j, k, n), Ipy(i, j, k, n), q,
                    vmac, pbc[n], dlo.y, dhi.y);
                Godunov_ppm_fpu_z_nolim(
                    i, j, k, n, l_dt, dz, Imz(i, j, k, n), Ipz(i, j, k, n), q,
                    wmac, pbc[n], dlo.z, dhi.z);
            });
        break;
    }
    case godunov::scheme::WENOJS: {
        amrex::ParallelFor(
            bxg1, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_weno_fpu_x(
                    i, j, k, n, l_dt, dx, Imx(i, j, k, n), Ipx(i, j, k, n), q,
                    umac, pbc[n], dlo.x, dhi.x, true);
                Godunov_weno_fpu_y(
                    i, j, k, n, l_dt, dy, Imy(i, j, k, n), Ipy(i, j, k, n), q,
                    vmac, pbc[n], dlo.y, dhi.y, true);
                Godunov_weno_fpu_z(
                    i, j, k, n, l_dt, dz, Imz(i, j, k, n), Ipz(i, j, k, n), q,
                    wmac, pbc[n], dlo.z, dhi.z, true);
            });
        break;
    }
    case godunov::scheme::WENOZ: {
        amrex::ParallelFor(
            bxg1, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_weno_fpu_x(
                    i, j, k, n, l_dt, dx, Imx(i, j, k, n), Ipx(i, j, k, n), q,
                    umac, pbc[n], dlo.x, dhi.x, false);
                Godunov_weno_fpu_y(
                    i, j, k, n, l_dt, dy, Imy(i, j, k, n), Ipy(i, j, k, n), q,
                    vmac, pbc[n], dlo.y, dhi.y, false);
                Godunov_weno_fpu_z(
                    i, j, k, n, l_dt, dz, Imz(i, j, k, n), Ipz(i, j, k, n), q,
                    wmac, pbc[n], dlo.z, dhi.z, false);
            });
        break;
    }
    case godunov::scheme::PLM: {
        amrex::ParallelFor(
            xebox, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_plm_fpu_x(
                    i, j, k, n, l_dt, dx, Imx(i, j, k, n), Ipx(i - 1, j, k, n),
                    q, umac(i, j, k), pbc[n], dlo.x, dhi.x);
            });

        amrex::ParallelFor(
            yebox, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_plm_fpu_y(
                    i, j, k, n, l_dt, dy, Imy(i, j, k, n), Ipy(i, j - 1, k, n),
                    q, vmac(i, j, k), pbc[n], dlo.y, dhi.y);
            });

        amrex::ParallelFor(
            zebox, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_plm_fpu_z(
                    i, j, k, n, l_dt, dz, Imz(i, j, k, n), Ipz(i, j, k - 1, n),
                    q, wmac(i, j, k), pbc[n], dlo.z, dhi.z);
            });
        break;
    }
    case godunov::scheme::MINMOD: {
        amrex::ParallelFor(
            bxg1, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_minmod_fpu_x(
                    i, j, k, n, l_dt, dx, Imx(i, j, k, n), Ipx(i, j, k, n), q,
                    umac, pbc[n], dlo.x, dhi.x);
                Godunov_minmod_fpu_y(
                    i, j, k, n, l_dt, dy, Imy(i, j, k, n), Ipy(i, j, k, n), q,
                    vmac, pbc[n], dlo.y, dhi.y);
                Godunov_minmod_fpu_z(
                    i, j, k, n, l_dt, dz, Imz(i, j, k, n), Ipz(i, j, k, n), q,
                    wmac, pbc[n], dlo.z, dhi.z);
            });
        break;
    }
    }

    amrex::ParallelFor(
        xebox, ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            constexpr Real small_vel = 1.e-10;

            Real uad = umac(i, j, k);
            Real fux = (amrex::Math::abs(uad) < small_vel) ? 0. : 1.;
            bool uval = uad >= 0.;
            // divu = 0
            // Real cons1 = (iconserv[n]) ? -0.5*l_dt*q(i-1,j,k,n)*divu(i-1,j,k)
            // : 0.; Real cons2 = (iconserv[n]) ? -0.5*l_dt*q(i  ,j,k,n)*divu(i
            // ,j,k) : 0.;
            Real lo = Ipx(i - 1, j, k, n); // + cons1;
            Real hi = Imx(i, j, k, n);     // + cons2;
            if (fq) {
                lo += 0.5 * l_dt * fq(i - 1, j, k, n);
                hi += 0.5 * l_dt * fq(i, j, k, n);
            }

            auto bc = pbc[n];

            Godunov_trans_xbc(
                i, j, k, n, q, lo, hi, uad, bc.lo(0), bc.hi(0), dlo.x, dhi.x);
            xlo(i, j, k, n) = lo;
            xhi(i, j, k, n) = hi;
            Real st = (uval) ? lo : hi;
            Imx(i, j, k, n) = fux * st + (1. - fux) * 0.5 * (hi + lo);
        },
        yebox, ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            constexpr Real small_vel = 1.e-10;

            Real vad = vmac(i, j, k);
            Real fuy = (amrex::Math::abs(vad) < small_vel) ? 0. : 1.;
            bool vval = vad >= 0.;
            // divu = 0
            // Real cons1 = (iconserv[n]) ? -0.5*l_dt*q(i,j-1,k,n)*divu(i,j-1,k)
            // : 0.; Real cons2 = (iconserv[n]) ? -0.5*l_dt*q(i,j ,k,n)*divu(i,j
            // ,k) : 0.;
            Real lo = Ipy(i, j - 1, k, n); // + cons1;
            Real hi = Imy(i, j, k, n);     // + cons2;
            if (fq) {
                lo += 0.5 * l_dt * fq(i, j - 1, k, n);
                hi += 0.5 * l_dt * fq(i, j, k, n);
            }

            auto bc = pbc[n];

            Godunov_trans_ybc(
                i, j, k, n, q, lo, hi, vad, bc.lo(1), bc.hi(1), dlo.y, dhi.y);

            ylo(i, j, k, n) = lo;
            yhi(i, j, k, n) = hi;
            Real st = (vval) ? lo : hi;
            Imy(i, j, k, n) = fuy * st + (1. - fuy) * 0.5 * (hi + lo);
        },
        zebox, ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            constexpr Real small_vel = 1.e-10;

            Real wad = wmac(i, j, k);
            Real fuz = (amrex::Math::abs(wad) < small_vel) ? 0. : 1.;
            bool wval = wad >= 0.;
            auto bc = pbc[n];
            // divu = 0
            // Real cons1 = (iconserv[n]) ? -0.5*l_dt*q(i,j,k-1,n)*divu(i,j,k-1)
            // : 0.; Real cons2 = (iconserv[n]) ? -0.5*l_dt*q(i,j,k
            // ,n)*divu(i,j,k  ) : 0.;
            Real lo = Ipz(i, j, k - 1, n); // + cons1;
            Real hi = Imz(i, j, k, n);     // + cons2;
            if (fq) {
                lo += 0.5 * l_dt * fq(i, j, k - 1, n);
                hi += 0.5 * l_dt * fq(i, j, k, n);
            }

            Godunov_trans_zbc(
                i, j, k, n, q, lo, hi, wad, bc.lo(2), bc.hi(2), dlo.z, dhi.z);

            zlo(i, j, k, n) = lo;
            zhi(i, j, k, n) = hi;
            Real st = (wval) ? lo : hi;
            Imz(i, j, k, n) = fuz * st + (1. - fuz) * 0.5 * (hi + lo);
        });

    Array4<Real> xedge = Imx;
    Array4<Real> yedge = Imy;
    Array4<Real> zedge = Imz;

    // We can reuse the space in Ipx, Ipy and Ipz.

    //
    // x-direction
    //
    Box const& xbxtmp = amrex::grow(bx, 0, 1);
    Array4<Real> yzlo =
        makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(xbxtmp, 1), ncomp);
    Array4<Real> zylo =
        makeArray4(xyzhi.dataPtr(), amrex::surroundingNodes(xbxtmp, 2), ncomp);
    amrex::ParallelFor(
        Box(zylo), ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const auto bc = pbc[n];
            Real l_zylo, l_zyhi;
            Godunov_corner_couple_zy(
                l_zylo, l_zyhi, i, j, k, n, l_dt, dy, iconserv[n] != 0,
                zlo(i, j, k, n), zhi(i, j, k, n), q, vmac, yedge);

            Real wad = wmac(i, j, k);
            Godunov_trans_zbc(
                i, j, k, n, q, l_zylo, l_zyhi, wad, bc.lo(2), bc.hi(2), dlo.z,
                dhi.z);

            constexpr Real small_vel = 1.e-10;

            Real st = (wad >= 0.) ? l_zylo : l_zyhi;
            Real fu = (amrex::Math::abs(wad) < small_vel) ? 0.0 : 1.0;
            zylo(i, j, k, n) = fu * st + (1.0 - fu) * 0.5 * (l_zyhi + l_zylo);
        },
        Box(yzlo), ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const auto bc = pbc[n];
            Real l_yzlo, l_yzhi;
            Godunov_corner_couple_yz(
                l_yzlo, l_yzhi, i, j, k, n, l_dt, dz, iconserv[n] != 0,
                ylo(i, j, k, n), yhi(i, j, k, n), q, wmac, zedge);

            Real vad = vmac(i, j, k);
            Godunov_trans_ybc(
                i, j, k, n, q, l_yzlo, l_yzhi, vad, bc.lo(1), bc.hi(1), dlo.y,
                dhi.y);

            constexpr Real small_vel = 1.e-10;

            Real st = (vad >= 0.) ? l_yzlo : l_yzhi;
            Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
            yzlo(i, j, k, n) = fu * st + (1.0 - fu) * 0.5 * (l_yzhi + l_yzlo);
        });
    //

    amrex::ParallelFor(
        xbx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real stl, sth;
            constexpr Real small_vel = 1.e-10;
            if (iconserv[n] != 0) {
                stl = xlo(i, j, k, n) -
                      (0.5 * dtdy) *
                          (yzlo(i - 1, j + 1, k, n) * vmac(i - 1, j + 1, k) -
                           yzlo(i - 1, j, k, n) * vmac(i - 1, j, k)) -
                      (0.5 * dtdz) *
                          (zylo(i - 1, j, k + 1, n) * wmac(i - 1, j, k + 1) -
                           zylo(i - 1, j, k, n) * wmac(i - 1, j, k)) +
                      (0.5 * dtdy) * q(i - 1, j, k, n) *
                          (vmac(i - 1, j + 1, k) - vmac(i - 1, j, k)) +
                      (0.5 * dtdz) * q(i - 1, j, k, n) *
                          (wmac(i - 1, j, k + 1) - wmac(i - 1, j, k));

                sth = xhi(i, j, k, n) -
                      (0.5 * dtdy) * (yzlo(i, j + 1, k, n) * vmac(i, j + 1, k) -
                                      yzlo(i, j, k, n) * vmac(i, j, k)) -
                      (0.5 * dtdz) * (zylo(i, j, k + 1, n) * wmac(i, j, k + 1) -
                                      zylo(i, j, k, n) * wmac(i, j, k)) +
                      (0.5 * dtdy) * q(i, j, k, n) *
                          (vmac(i, j + 1, k) - vmac(i, j, k)) +
                      (0.5 * dtdz) * q(i, j, k, n) *
                          (wmac(i, j, k + 1) - wmac(i, j, k));
            } else {
                stl = xlo(i, j, k, n) -
                      (0.25 * dtdy) *
                          (vmac(i - 1, j + 1, k) + vmac(i - 1, j, k)) *
                          (yzlo(i - 1, j + 1, k, n) - yzlo(i - 1, j, k, n)) -
                      (0.25 * dtdz) *
                          (wmac(i - 1, j, k + 1) + wmac(i - 1, j, k)) *
                          (zylo(i - 1, j, k + 1, n) - zylo(i - 1, j, k, n));

                sth = xhi(i, j, k, n) -
                      (0.25 * dtdy) * (vmac(i, j + 1, k) + vmac(i, j, k)) *
                          (yzlo(i, j + 1, k, n) - yzlo(i, j, k, n)) -
                      (0.25 * dtdz) * (wmac(i, j, k + 1) + wmac(i, j, k)) *
                          (zylo(i, j, k + 1, n) - zylo(i, j, k, n));
            }

            auto bc = pbc[n];
            Godunov_cc_xbc_lo(i, j, k, n, q, stl, sth, umac, bc.lo(0), dlo.x);
            Godunov_cc_xbc_hi(i, j, k, n, q, stl, sth, umac, bc.hi(0), dhi.x);

            Real qx = (umac(i, j, k) >= 0.) ? stl : sth;
            qx = (amrex::Math::abs(umac(i, j, k)) < small_vel)
                     ? 0.5 * (stl + sth)
                     : qx;

            bool save_flux = false;
            if (flag(i, j, k) == 1 || flag(i - 1, j, k) == 1) {
                save_flux = true;
            }

            if (save_flux) {
                if (iconserv[n] == 1) {
                    fx(i, j, k, n) = umac(i, j, k) * qx;
                } else {
                    if (iconserv[n] < 0) {
                        fx(i, j, k, 0) /= qx;
                        fx(i, j, k, 1) /= qx;
                        fx(i, j, k, 2) /= qx;
                    } else {
                        fx(i, j, k, n) = qx;
                    }
                }
            }
        });

    //
    // y-direction
    //
    Box const& ybxtmp = amrex::grow(bx, 1, 1);
    Array4<Real> xzlo =
        makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(ybxtmp, 0), ncomp);
    Array4<Real> zxlo =
        makeArray4(xyzhi.dataPtr(), amrex::surroundingNodes(ybxtmp, 2), ncomp);
    amrex::ParallelFor(
        Box(xzlo), ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const auto bc = pbc[n];
            Real l_xzlo, l_xzhi;
            Godunov_corner_couple_xz(
                l_xzlo, l_xzhi, i, j, k, n, l_dt, dz, iconserv[n] != 0,
                xlo(i, j, k, n), xhi(i, j, k, n), q, wmac, zedge);

            Real uad = umac(i, j, k);
            Godunov_trans_xbc(
                i, j, k, n, q, l_xzlo, l_xzhi, uad, bc.lo(0), bc.hi(0), dlo.x,
                dhi.x);

            constexpr Real small_vel = 1.e-10;

            Real st = (uad >= 0.) ? l_xzlo : l_xzhi;
            Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
            xzlo(i, j, k, n) = fu * st + (1.0 - fu) * 0.5 * (l_xzhi + l_xzlo);
        },
        Box(zxlo), ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const auto bc = pbc[n];
            Real l_zxlo, l_zxhi;
            Godunov_corner_couple_zx(
                l_zxlo, l_zxhi, i, j, k, n, l_dt, dx, iconserv[n] != 0,
                zlo(i, j, k, n), zhi(i, j, k, n), q, umac, xedge);

            Real wad = wmac(i, j, k);
            Godunov_trans_zbc(
                i, j, k, n, q, l_zxlo, l_zxhi, wad, bc.lo(2), bc.hi(2), dlo.z,
                dhi.z);

            constexpr Real small_vel = 1.e-10;

            Real st = (wad >= 0.) ? l_zxlo : l_zxhi;
            Real fu = (amrex::Math::abs(wad) < small_vel) ? 0.0 : 1.0;
            zxlo(i, j, k, n) = fu * st + (1.0 - fu) * 0.5 * (l_zxhi + l_zxlo);
        });

    amrex::ParallelFor(
        ybx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real stl, sth;
            constexpr Real small_vel = 1.e-10;

            if (iconserv[n] != 0) {
                stl = ylo(i, j, k, n) -
                      (0.5 * dtdx) *
                          (xzlo(i + 1, j - 1, k, n) * umac(i + 1, j - 1, k) -
                           xzlo(i, j - 1, k, n) * umac(i, j - 1, k)) -
                      (0.5 * dtdz) *
                          (zxlo(i, j - 1, k + 1, n) * wmac(i, j - 1, k + 1) -
                           zxlo(i, j - 1, k, n) * wmac(i, j - 1, k)) +
                      (0.5 * dtdx) * q(i, j - 1, k, n) *
                          (umac(i + 1, j - 1, k) - umac(i, j - 1, k)) +
                      (0.5 * dtdz) * q(i, j - 1, k, n) *
                          (wmac(i, j - 1, k + 1) - wmac(i, j - 1, k));

                sth = yhi(i, j, k, n) -
                      (0.5 * dtdx) * (xzlo(i + 1, j, k, n) * umac(i + 1, j, k) -
                                      xzlo(i, j, k, n) * umac(i, j, k)) -
                      (0.5 * dtdz) * (zxlo(i, j, k + 1, n) * wmac(i, j, k + 1) -
                                      zxlo(i, j, k, n) * wmac(i, j, k)) +
                      (0.5 * dtdx) * q(i, j, k, n) *
                          (umac(i + 1, j, k) - umac(i, j, k)) +
                      (0.5 * dtdz) * q(i, j, k, n) *
                          (wmac(i, j, k + 1) - wmac(i, j, k));
            } else {
                stl = ylo(i, j, k, n) -
                      (0.25 * dtdx) *
                          (umac(i + 1, j - 1, k) + umac(i, j - 1, k)) *
                          (xzlo(i + 1, j - 1, k, n) - xzlo(i, j - 1, k, n)) -
                      (0.25 * dtdz) *
                          (wmac(i, j - 1, k + 1) + wmac(i, j - 1, k)) *
                          (zxlo(i, j - 1, k + 1, n) - zxlo(i, j - 1, k, n));

                sth = yhi(i, j, k, n) -
                      (0.25 * dtdx) * (umac(i + 1, j, k) + umac(i, j, k)) *
                          (xzlo(i + 1, j, k, n) - xzlo(i, j, k, n)) -
                      (0.25 * dtdz) * (wmac(i, j, k + 1) + wmac(i, j, k)) *
                          (zxlo(i, j, k + 1, n) - zxlo(i, j, k, n));
            }

            auto bc = pbc[n];
            Godunov_cc_ybc_lo(i, j, k, n, q, stl, sth, vmac, bc.lo(1), dlo.y);
            Godunov_cc_ybc_hi(i, j, k, n, q, stl, sth, vmac, bc.hi(1), dhi.y);

            Real qy = (vmac(i, j, k) >= 0.) ? stl : sth;
            qy = (amrex::Math::abs(vmac(i, j, k)) < small_vel)
                     ? 0.5 * (stl + sth)
                     : qy;

            bool save_flux = false;
            if (flag(i, j, k) == 1 || flag(i, j - 1, k) == 1) {
                save_flux = true;
            }

            if (save_flux) {
                if (iconserv[n] == 1) {
                    fy(i, j, k, n) = vmac(i, j, k) * qy;
                } else {
                    if (iconserv[n] < 0) {
                        fy(i, j, k, 0) /= qy;
                        fy(i, j, k, 1) /= qy;
                        fy(i, j, k, 2) /= qy;
                    } else {
                        fy(i, j, k, n) = qy;
                    }
                }
            }
        });

    //
    // z-direction
    //
    Box const& zbxtmp = amrex::grow(bx, 2, 1);
    Array4<Real> xylo =
        makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(zbxtmp, 0), ncomp);
    Array4<Real> yxlo =
        makeArray4(xyzhi.dataPtr(), amrex::surroundingNodes(zbxtmp, 1), ncomp);
    amrex::ParallelFor(
        Box(xylo), ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const auto bc = pbc[n];
            Real l_xylo, l_xyhi;
            Godunov_corner_couple_xy(
                l_xylo, l_xyhi, i, j, k, n, l_dt, dy, iconserv[n] != 0,
                xlo(i, j, k, n), xhi(i, j, k, n), q, vmac, yedge);

            Real uad = umac(i, j, k);
            Godunov_trans_xbc(
                i, j, k, n, q, l_xylo, l_xyhi, uad, bc.lo(0), bc.hi(0), dlo.x,
                dhi.x);

            constexpr Real small_vel = 1.e-10;

            Real st = (uad >= 0.) ? l_xylo : l_xyhi;
            Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
            xylo(i, j, k, n) = fu * st + (1.0 - fu) * 0.5 * (l_xyhi + l_xylo);
        },
        Box(yxlo), ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const auto bc = pbc[n];
            Real l_yxlo, l_yxhi;
            Godunov_corner_couple_yx(
                l_yxlo, l_yxhi, i, j, k, n, l_dt, dx, iconserv[n] != 0,
                ylo(i, j, k, n), yhi(i, j, k, n), q, umac, xedge);

            Real vad = vmac(i, j, k);
            Godunov_trans_ybc(
                i, j, k, n, q, l_yxlo, l_yxhi, vad, bc.lo(1), bc.hi(1), dlo.y,
                dhi.y);

            constexpr Real small_vel = 1.e-10;

            Real st = (vad >= 0.) ? l_yxlo : l_yxhi;
            Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
            yxlo(i, j, k, n) = fu * st + (1.0 - fu) * 0.5 * (l_yxhi + l_yxlo);
        });

    amrex::ParallelFor(
        zbx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real stl, sth;
            constexpr Real small_vel = 1.e-10;
            if (iconserv[n] != 0) {
                stl = zlo(i, j, k, n) -
                      (0.5 * dtdx) *
                          (xylo(i + 1, j, k - 1, n) * umac(i + 1, j, k - 1) -
                           xylo(i, j, k - 1, n) * umac(i, j, k - 1)) -
                      (0.5 * dtdy) *
                          (yxlo(i, j + 1, k - 1, n) * vmac(i, j + 1, k - 1) -
                           yxlo(i, j, k - 1, n) * vmac(i, j, k - 1)) +
                      (0.5 * dtdx) * q(i, j, k - 1, n) *
                          (umac(i + 1, j, k - 1) - umac(i, j, k - 1)) +
                      (0.5 * dtdy) * q(i, j, k - 1, n) *
                          (vmac(i, j + 1, k - 1) - vmac(i, j, k - 1));

                sth = zhi(i, j, k, n) -
                      (0.5 * dtdx) * (xylo(i + 1, j, k, n) * umac(i + 1, j, k) -
                                      xylo(i, j, k, n) * umac(i, j, k)) -
                      (0.5 * dtdy) * (yxlo(i, j + 1, k, n) * vmac(i, j + 1, k) -
                                      yxlo(i, j, k, n) * vmac(i, j, k)) +
                      (0.5 * dtdx) * q(i, j, k, n) *
                          (umac(i + 1, j, k) - umac(i, j, k)) +
                      (0.5 * dtdy) * q(i, j, k, n) *
                          (vmac(i, j + 1, k) - vmac(i, j, k));
            } else {
                stl = zlo(i, j, k, n) -
                      (0.25 * dtdx) *
                          (umac(i + 1, j, k - 1) + umac(i, j, k - 1)) *
                          (xylo(i + 1, j, k - 1, n) - xylo(i, j, k - 1, n)) -
                      (0.25 * dtdy) *
                          (vmac(i, j + 1, k - 1) + vmac(i, j, k - 1)) *
                          (yxlo(i, j + 1, k - 1, n) - yxlo(i, j, k - 1, n));

                sth = zhi(i, j, k, n) -
                      (0.25 * dtdx) * (umac(i + 1, j, k) + umac(i, j, k)) *
                          (xylo(i + 1, j, k, n) - xylo(i, j, k, n)) -
                      (0.25 * dtdy) * (vmac(i, j + 1, k) + vmac(i, j, k)) *
                          (yxlo(i, j + 1, k, n) - yxlo(i, j, k, n));
            }

            auto bc = pbc[n];
            Godunov_cc_zbc_lo(i, j, k, n, q, stl, sth, wmac, bc.lo(2), dlo.z);
            Godunov_cc_zbc_hi(i, j, k, n, q, stl, sth, wmac, bc.hi(2), dhi.z);

            Real qz = (wmac(i, j, k) >= 0.) ? stl : sth;
            qz = (amrex::Math::abs(wmac(i, j, k)) < small_vel)
                     ? 0.5 * (stl + sth)
                     : qz;

            bool save_flux = false;
            if (flag(i, j, k) == 1 || flag(i, j, k - 1) == 1) {
                save_flux = true;
            }

            if (save_flux) {
                if (iconserv[n] == 1) {
                    fz(i, j, k, n) = wmac(i, j, k) * qz;
                } else {
                    if (iconserv[n] < 0) {
                        fz(i, j, k, 0) /= qz;
                        fz(i, j, k, 1) /= qz;
                        fz(i, j, k, 2) /= qz;
                    } else {
                        fz(i, j, k, n) = qz;
                    }
                }
            }
        });
}

void godunov::compute_fluxes(
    int lev,
    Box const& bx,
    int ncomp,
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz,
    Array4<Real const> const& q,
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac,
    Array4<Real const> const& fq,
    BCRec const* pbc,
    int const* iconserv,
    Real* p,
    Vector<amrex::Geometry> geom,
    Real dt,
    godunov::scheme godunov_scheme)
{
    amrex::IArrayBox tmpfab(grow(bx, 2), 1);
    Array4<int> flag_all = makeArray4(tmpfab.dataPtr(), grow(bx, 2), 1);
    amrex::ParallelFor(
        grow(bx, 2), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            flag_all(i, j, k) = 1;
        });

    godunov::compute_fluxes(
        lev, bx, ncomp, fx, fy, fz, q, umac, vmac, wmac, fq, pbc, iconserv, p,
        geom, dt, godunov_scheme, flag_all);
}

void godunov::compute_advection(
    int lev,
    Box const& bx,
    int ncomp,
    Array4<Real> const& dqdt,
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz,
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac,
    int const* iconserv,
    Vector<amrex::Geometry> geom)
{

    BL_PROFILE("amr-wind::godunov::compute_advection");

    const auto dxinv = geom[lev].InvCellSizeArray();

    amrex::ParallelFor(
        bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            if (iconserv[n] != 0) {
                dqdt(i, j, k, n) =
                    dxinv[0] * (fx(i, j, k, n) - fx(i + 1, j, k, n)) +
                    dxinv[1] * (fy(i, j, k, n) - fy(i, j + 1, k, n)) +
                    dxinv[2] * (fz(i, j, k, n) - fz(i, j, k + 1, n));
            } else {
                dqdt(i, j, k, n) =
                    0.5 * dxinv[0] * (umac(i, j, k) + umac(i + 1, j, k)) *
                        (fx(i, j, k, n) - fx(i + 1, j, k, n)) +
                    0.5 * dxinv[1] * (vmac(i, j, k) + vmac(i, j + 1, k)) *
                        (fy(i, j, k, n) - fy(i, j + 1, k, n)) +
                    0.5 * dxinv[2] * (wmac(i, j, k) + wmac(i, j, k + 1)) *
                        (fz(i, j, k, n) - fz(i, j, k + 1, n));
            }
        });
}

void godunov::multiphase_fluxes(
    int lev,
    Box const& bx,
    int ncomp,
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz,
    Array4<Real const> const& q,
    Array4<Real const> const& aa_x,
    Array4<Real const> const& aa_y,
    Array4<Real const> const& aa_z,
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac,
    Array4<Real const> const& fq,
    Array4<Real const> const& vofn,
    BCRec const* pbc,
    Real* p,
    Vector<amrex::Geometry> geom,
    Real dt,
    Real rho1,
    Real rho2,
    godunov::scheme mflux_scheme,
    Array4<int const> const& flag)
{

    BL_PROFILE("amr-wind::godunov::compute_fluxes");
    Box const& xbx = amrex::surroundingNodes(bx, 0);
    Box const& ybx = amrex::surroundingNodes(bx, 1);
    Box const& zbx = amrex::surroundingNodes(bx, 2);
    Box const& bxg1 = amrex::grow(bx, 1);
    Box const& bxg2 = amrex::grow(bx, 2);

    const Real dx = geom[lev].CellSize(0);
    const Real dy = geom[lev].CellSize(1);
    const Real dz = geom[lev].CellSize(2);

    Box const& domain = geom[lev].Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);

    Array4<Real> Gu = makeArray4(p, bxg1, ncomp);
    p += Gu.size();
    Array4<Real> Gv = makeArray4(p, bxg1, ncomp);
    p += Gv.size();
    Array4<Real> Gw = makeArray4(p, bxg1, ncomp);
    p += Gw.size();

    // Generate cell-centered gradients (should add forces)
    switch (mflux_scheme) {
    case godunov::scheme::MINMOD: {
        amrex::ParallelFor(
            bxg1, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Godunov_minmod_cc_grad(
                    i, j, k, n, dx, dy, dz, q, Gu(i, j, k, n), Gv(i, j, k, n),
                    Gw(i, j, k, n));
            });
        break;
    }
    }

    // Interpolate momentum and density to faces, then
    // Save fluxes where they should be saved

    // X-direction
    amrex::ParallelFor(
        xbx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real xf = dt * umac(i, j, k);
            // Select index and sign of spatial interpolation (upwinding)
            bool upw = xf > 0.;
            int ii = upw ? i - 1 : i;
            Real sdir = upw ? 1. : -1.;
            // Interpolate other MAC velocities to face
            Real yf = dt * 0.5 * (vmac(ii, j, k) + vmac(ii, j + 1, k));
            Real zf = dt * 0.5 * (wmac(ii, j, k) + wmac(ii, j, k + 1));
            // Map normal displacements according to phase,
            // resort to upwinding if phase was not present before
            Real xf_l = (vofn(ii, j, k) * dx < aa_x(i, j, k) * amrex::Math::abs(xf))
                            ? sdir * dx
                            : xf * aa_x(i, j, k) / (vofn(ii, j, k) + 1e-15);
            Real xf_g = ((1.0 - vofn(ii, j, k)) * dx <
                         (1.0 - aa_x(i, j, k)) * amrex::Math::abs(xf))
                            ? sdir * dx
                            : xf * (1.0 - aa_x(i, j, k)) /
                                  (1.0 - vofn(ii, j, k) + 1e-15);
            // First gradient will be used for spatial interpolation
            // Liquid phase, then gas phase
            Real uf_l = sptemp_interp(
                q(ii, j, k, n), sdir * dx, xf_l, yf, zf, Gu(ii, j, k, n),
                Gv(ii, j, k, n), Gw(ii, j, k, n));
            Real uf_g = sptemp_interp(
                q(ii, j, k, n), sdir * dx, xf_g, yf, zf, Gu(ii, j, k, n),
                Gv(ii, j, k, n), Gw(ii, j, k, n));
            // Check if flux should be saved
            if (flag(i, j, k) == 1 || flag(i - 1, j, k) == 1) {
                fx(i, j, k, n) =
                    umac(i, j, k) *
                    (rho1 * uf_l * aa_x(i, j, k) +
                     rho2 * uf_g * (1.0 - aa_x(i, j, k))) /
                    (rho1 * aa_x(i, j, k) + rho2 * (1.0 - aa_x(i, j, k)));
            }
        });
    // Y-direction
    amrex::ParallelFor(
        ybx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real yf = dt * vmac(i, j, k);
            // Select index and sign of spatial interpolation (upwinding)
            bool upw = yf > 0.;
            int jj = upw ? j - 1 : j;
            Real sdir = upw ? 1. : -1.;
            // Interpolate other MAC velocities to face
            Real xf = dt * 0.5 * (umac(i, jj, k) + umac(i + 1, jj, k));
            Real zf = dt * 0.5 * (wmac(i, jj, k) + wmac(i, jj, k + 1));
            // Map normal displacements according to phase,
            // resort to upwinding if phase was not present before
            Real yf_l = (vofn(i, jj, k) * dy < aa_y(i, j, k) * amrex::Math::abs(yf))
                            ? sdir * dy
                            : yf * aa_y(i, j, k) / (vofn(i, jj, k) + 1e-15);
            Real yf_g = ((1.0 - vofn(i, jj, k)) * dy <
                         (1.0 - aa_y(i, j, k)) * amrex::Math::abs(yf))
                            ? sdir * dy
                            : yf * (1.0 - aa_y(i, j, k)) /
                                  (1.0 - vofn(i, jj, k) + 1e-15);
            // First gradient will be used for spatial interpolation
            Real vf_l = sptemp_interp(
                q(i, jj, k, n), sdir * dy, yf_l, xf, zf, Gv(i, jj, k, n),
                Gu(i, jj, k, n), Gw(i, jj, k, n));
            Real vf_g = sptemp_interp(
                q(i, jj, k, n), sdir * dy, yf_g, xf, zf, Gv(i, jj, k, n),
                Gu(i, jj, k, n), Gw(i, jj, k, n));
            // Check if flux should be saved
            if (flag(i, j, k) == 1 || flag(i, j - 1, k) == 1) {
                fy(i, j, k, n) =
                    vmac(i, j, k) *
                    (rho1 * vf_l * aa_y(i, j, k) +
                     rho2 * vf_g * (1.0 - aa_y(i, j, k))) /
                    (rho1 * aa_y(i, j, k) + rho2 * (1.0 - aa_y(i, j, k)));
            }
        });
    // Z-direction
    amrex::ParallelFor(
        zbx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real zf = dt * wmac(i, j, k);
            // Select index and sign of spatial interpolation (upwinding)
            bool upw = zf > 0.;
            int kk = upw ? k - 1 : k;
            Real sdir = upw ? 1. : -1.;
            // Interpolate other MAC velocities to face
            Real xf = dt * 0.5 * (umac(i, j, kk) + umac(i + 1, j, kk));
            Real yf = dt * 0.5 * (vmac(i, j, kk) + vmac(i, j + 1, kk));
            // Map normal displacements according to phase,
            // resort to upwinding if phase was not present before
            Real zf_l = (vofn(i, j, kk) * dz < aa_z(i, j, k) * amrex::Math::abs(zf))
                            ? sdir * dz
                            : zf * aa_z(i, j, k) / (vofn(i, j, kk) + 1e-15);
            Real zf_g = ((1.0 - vofn(i, j, kk)) * dz <
                         (1.0 - aa_z(i, j, k)) * amrex::Math::abs(zf))
                            ? sdir * dz
                            : zf * (1.0 - aa_z(i, j, k)) /
                                  ((1.0 - vofn(i, j, kk)) + 1e-15);
            // First gradient will be used for spatial interpolation
            Real wf_l = sptemp_interp(
                q(i, j, kk, n), sdir * dz, zf_l, xf, yf, Gw(i, j, kk, n),
                Gu(i, j, kk, n), Gv(i, j, kk, n));
            Real wf_g = sptemp_interp(
                q(i, j, kk, n), sdir * dz, zf_g, xf, yf, Gw(i, j, kk, n),
                Gu(i, j, kk, n), Gv(i, j, kk, n));
            // Check if flux should be saved
            if (flag(i, j, k) == 1 || flag(i, j, k - 1) == 1) {
                fz(i, j, k, n) =
                    wmac(i, j, k) *
                    (rho1 * wf_l * aa_z(i, j, k) +
                     rho2 * wf_g * (1.0 - aa_z(i, j, k))) /
                    (rho1 * aa_z(i, j, k) + rho2 * (1.0 - aa_z(i, j, k)));
            }
        });
}

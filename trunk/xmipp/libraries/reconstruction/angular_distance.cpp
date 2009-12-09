/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano      coss@cnb.csic.es (2002)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "angular_distance.h"

#include <data/args.h>
#include <data/histogram.h>

// Read arguments ==========================================================
void Prog_angular_distance_prm::read(int argc, char **argv)
{
    fn_ang1 = getParameter(argc, argv, "-ang1");
    fn_ang2 = getParameter(argc, argv, "-ang2");
    fn_ang_out = getParameter(argc, argv, "-o", "");
    fn_sym = getParameter(argc, argv, "-sym", "");
    check_mirrors = checkParameter(argc, argv, "-check_mirrors");
    object_rotation = checkParameter(argc, argv, "-object_rotation");
}

// Show ====================================================================
void Prog_angular_distance_prm::show()
{
    std::cout << "Angular docfile 1: " << fn_ang1       << std::endl
              << "Angular docfile 2: " << fn_ang2       << std::endl
              << "Angular output   : " << fn_ang_out    << std::endl
              << "Symmetry file    : " << fn_sym        << std::endl
              << "Check mirrors    : " << check_mirrors << std::endl
              << "Object rotation  : " << object_rotation<<std::endl
    ;
}

// usage ===================================================================
void Prog_angular_distance_prm::usage()
{
    std::cerr << "   -ang1 <DocFile 1>         : Angular document file 1\n"
              << "   -ang2 <DocFile 2>         : Angular document file 2\n"
              << "  [-o <DocFile out>]         : Merge dcfile. If not given it is\n"
              << "                               not generated\n"
              << "  [-sym <symmetry file>]     : Symmetry file if any\n"
              << "  [-check_mirrors]           : Check if mirrored axes give better\n"
              << "  [-object_rotation]         : Use object rotations rather than projection directions\n"
              << "                               fit (Spider, APMQ)\n"
    ;
}

// Produce side information ================================================
void Prog_angular_distance_prm::produce_side_info()
{
    if (fn_ang1 != "") DF1.read(fn_ang1);
    if (fn_ang2 != "") DF2.read(fn_ang2);
    if (fn_sym != "") SL.read_sym_file(fn_sym);

    // Check that both docfiles are of the same length
    if (DF1.dataLineNo() != DF2.dataLineNo())
        REPORT_ERROR(1, "Angular_distance: Input Docfiles with different number of entries");
}

// 2nd angle set -----------------------------------------------------------
#define SHOW_ANGLES(rot,tilt,psi) \
    std::cout << #rot  << "=" << rot << " " \
    << #tilt << "=" << tilt << " " \
    << #psi  << "=" << psi << " ";
#define DEBUG
double Prog_angular_distance_prm::second_angle_set(double rot1, double tilt1,
        double psi1, double &rot2, double &tilt2, double &psi2,
        bool projdir_mode)
{
#ifdef DEBUG
    std::cout << "   ";
    SHOW_ANGLES(rot2, tilt2, psi2);
#endif

    // Distance based on Euler axes
    Matrix2D<double> E1, E2;
    Euler_angles2matrix(rot1, tilt1, psi1, E1);
    Euler_angles2matrix(rot2, tilt2, psi2, E2);
    Matrix1D<double> v1, v2;
    double axes_dist = 0;
    double N = 0;
    for (int i = 0; i < 3; i++)
    {
        if (projdir_mode && i != 2) continue;
        E1.getRow(i, v1);
        E2.getRow(i, v2);
        double dist = RAD2DEG(acos(CLIP(dotProduct(v1, v2), -1, 1)));
        axes_dist += dist;
        N++;
#ifdef DEBUG
        std::cout << "d(" << i << ")=" << dist << " ";
#endif
    }
    axes_dist /= N;


#ifdef DEBUG
    std::cout << "-->" << axes_dist << std::endl;
#endif

    return axes_dist;
}
#undef DEBUG

// Check symmetries --------------------------------------------------------
//#define DEBUG
double Prog_angular_distance_prm::check_symmetries(double rot1, double tilt1,
        double psi1, double &rot2, double &tilt2, double &psi2,
        bool projdir_mode)
{
#ifdef DEBUG
    SHOW_ANGLES(rot1, tilt1, psi1);
    std::cout << std::endl;
#endif

    int imax = SL.SymsNo() + 1;
    Matrix2D<double>  L(4, 4), R(4, 4);  // A matrix from the list
    double best_ang_dist = 3600;
    double best_rot2, best_tilt2, best_psi2;

    for (int i = 0; i < imax; i++)
    {
        double rot2p, tilt2p, psi2p;
        if (i == 0)
        {
            rot2p = rot2;
            tilt2p = tilt2;
            psi2p = psi2;
        }
        else
        {
            SL.get_matrices(i - 1, L, R);
            L.resize(3, 3); // Erase last row and column
            R.resize(3, 3); // as only the relative orientation
            // is useful and not the translation
            if (object_rotation)
                Euler_apply_transf(R, L, rot2, tilt2, psi2, rot2p, tilt2p, psi2p);
            else
                Euler_apply_transf(L, R, rot2, tilt2, psi2, rot2p, tilt2p, psi2p);
        }

        double ang_dist = second_angle_set(rot1, tilt1, psi1, rot2p, tilt2p, psi2p,
                                           projdir_mode);

        if (ang_dist < best_ang_dist)
        {
            best_rot2 = rot2p;
            best_tilt2 = tilt2p;
            best_psi2 = psi2p;
            best_ang_dist = ang_dist;
        }

        if (check_mirrors)
        {
            Euler_up_down(rot2p, tilt2p, psi2p, rot2p, tilt2p, psi2p);
            double ang_dist_mirror = second_angle_set(rot1, tilt1, psi1, rot2p, tilt2p, psi2p,
                                     projdir_mode);

            if (ang_dist_mirror < best_ang_dist)
            {
                best_rot2 = rot2p;
                best_tilt2 = tilt2p;
                best_psi2 = psi2p;
                best_ang_dist = ang_dist_mirror;
            }

        }
    }
#ifdef DEBUG
    std::cout << "   Best=" << best_ang_dist << std::endl;
#endif
    rot2 = best_rot2;
    tilt2 = best_tilt2;
    psi2 = best_psi2;
    return best_ang_dist;
}
#define DEBUG

// Compute distance --------------------------------------------------------
void Prog_angular_distance_prm::compute_distance(double &angular_distance,
    double &shift_distance)
{
    DocFile DF_out;
    angular_distance = 0;
    shift_distance = 0;

    DF1.go_first_data_line();
    DF2.go_first_data_line();
    DF_out.reserve(DF1.dataLineNo());
    
    int dim = DF1.FirstLine_colNumber();
    Matrix1D<double> aux(17);
    Matrix1D<double> rot_diff, tilt_diff, psi_diff, vec_diff,
        X_diff, Y_diff, shift_diff;
    rot_diff.resize(DF1.dataLineNo());
    tilt_diff.resize(rot_diff);
    psi_diff.resize(rot_diff);
    vec_diff.resize(rot_diff);
    X_diff.resize(rot_diff);
    Y_diff.resize(rot_diff);
    shift_diff.resize(rot_diff);

    // Build output comment
    DF_out.append_comment("            rot1       rot2    diff_rot     tilt1      tilt2    diff_tilt    psi1       psi2     diff_psi   ang_dist      X1         X2        Xdiff       Y1          Y2       Ydiff     ShiftDiff");

    int i = 0;
    while (!DF1.eof())
    {
        // Read input data
        double rot1,  tilt1,  psi1;
        double rot2,  tilt2,  psi2;
        double rot2p, tilt2p, psi2p;
        double distp;
        double X1, X2, Y1, Y2;
        if (dim >= 1)
        {
            rot1 = DF1(0);
            rot2 = DF2(0);
        }
        else
        {
            rot1 = rot2 = 0;
        }
        if (dim >= 2)
        {
            tilt1 = DF1(1);
            tilt2 = DF2(1);
        }
        else
        {
            tilt1 = tilt2 = 0;
        }
        if (dim >= 3)
        {
            psi1 = DF1(2);
            psi2 = DF2(2);
        }
        else
        {
            psi1 = psi2 = 0;
        }
        if (dim >= 4)
        {
            X1 = DF1(3);
            X2 = DF2(3);
        }
        else
        {
            X1 = X2 = 0;
        }
        if (dim >= 5)
        {
            Y1 = DF1(4);
            Y2 = DF2(4);
        }
        else
        {
            Y1 = Y2 = 0;
        }

        // Bring both angles to a normalized set
        rot1 = realWRAP(rot1, -180, 180);
        tilt1 = realWRAP(tilt1, -180, 180);
        psi1 = realWRAP(psi1, -180, 180);

        rot2 = realWRAP(rot2, -180, 180);
        tilt2 = realWRAP(tilt2, -180, 180);
        psi2 = realWRAP(psi2, -180, 180);

        // Apply rotations to find the minimum distance angles
        rot2p = rot2;
        tilt2p = tilt2;
        psi2p = psi2;
        distp = check_symmetries(rot1, tilt1, psi1, rot2p, tilt2p, psi2p);

        // Compute angular difference
        rot_diff(i) = rot1 - rot2p;
        tilt_diff(i) = tilt1 - tilt2p;
        psi_diff(i) = psi1 - psi2p;
        vec_diff(i) = distp;
        X_diff(i) = X1 - X2;
        Y_diff(i) = Y1 - Y2;
        shift_diff(i) = sqrt(X_diff(i)*X_diff(i)+Y_diff(i)*Y_diff(i));

        // Fill the output result
        aux(0) = rot1;
        aux(1) = rot2p;
        aux(2) = rot_diff(i);
        aux(3) = tilt1;
        aux(4) = tilt2p;
        aux(5) = tilt_diff(i);
        aux(6) = psi1;
        aux(7) = psi2p;
        aux(8) = psi_diff(i);
        aux(9) = distp;
        aux(10) = X1;
        aux(11) = X2;
        aux(12) = X_diff(i);
        aux(13) = Y1;
        aux(14) = Y2;
        aux(15) = Y_diff(i);
        aux(16) = shift_diff(i);
        angular_distance += distp;
        shift_distance += shift_diff(i);
        DF_out.append_data_line(aux);

        // Move to next data line
        DF1.next_data_line();
        DF2.next_data_line();
        i++;
    }
    angular_distance /= i;
    shift_distance /=i;

    if (fn_ang_out != "")
    {
        DF_out.write(fn_ang_out + "_merge.txt");
        histogram1D hist;
        compute_hist(rot_diff, hist, 100);
        hist.write(fn_ang_out + "_rot_diff_hist.txt");
        compute_hist(tilt_diff, hist, 100);
        hist.write(fn_ang_out + "_tilt_diff_hist.txt");
        compute_hist(psi_diff, hist, 100);
        hist.write(fn_ang_out + "_psi_diff_hist.txt");
        compute_hist(vec_diff, hist, 0, 180, 180);
        hist.write(fn_ang_out + "_vec_diff_hist.txt");
        compute_hist(X_diff, hist, 20);
        hist.write(fn_ang_out + "_X_diff_hist.txt");
        compute_hist(Y_diff, hist, 20);
        hist.write(fn_ang_out + "_Y_diff_hist.txt");
        compute_hist(shift_diff, hist, 20);
        hist.write(fn_ang_out + "_shift_diff_hist.txt");
    }
}

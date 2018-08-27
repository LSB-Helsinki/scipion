import math
import sys
import random
import numpy as np
from itertools import izip
from glob import glob

from matrix3 import *
from vector3 import *
from euler import *
from os.path import splitext
from os.path import basename

from pyrelion import MetaData
import pyworkflow.utils as pwutils
from pyworkflow.utils.path import moveTree
from pyworkflow.em.transformations import vector_norm, unit_vector


class Vector3:
    def __init__(self):
        self.vector = np.empty(3, dtype=float)
        self.length = 0

    def set_vector(self, v):
        self.vector = np.array(v)

    def set_length(self, d):
        self.length = float(d)

    def compute_length(self):
        self.set_vector(vector_norm(self.vector))

    def compute_unit_vector(self):
        self.set_vector(unit_vector(self.vector))


def load_vectors(cmm_file, vectors_str, distances_str, angpix):
    """ Load subparticle vectors either from Chimera CMM file or from
    a vectors string. Distances can also be specified for each vector
    in the distances_str. """

    if cmm_file:
        subparticle_vector_list = vectors_from_cmm(cmm_file, angpix)
    else:
        subparticle_vector_list = vectors_from_string(vectors_str)

    if distances_str:
        # Change distances from A to pixel units
        subparticle_distances = [float(x) / angpix for x in
                                 distances_str.split(',')]

        if len(subparticle_distances) != len(subparticle_vector_list):
            raise Exception("Error: The number of distances does not match "
                            "the number of vectors!")

        for vector, distance in izip(subparticle_vector_list,
                                     subparticle_distances):
            if distance > 0:
                vector.set_length(distance)
            else:
                vector.compute_length()
    else:
        for vector in subparticle_vector_list:
            vector.compute_length()

    print "Using vectors:"

    for subparticle_vector in subparticle_vector_list:
        print "Vector: ",
        subparticle_vector.normalize()
        subparticle_vector.compute_matrix()
        subparticle_vector.print_vector()
        print ""
        print "Length: %.2f pixels" % subparticle_vector.distance()
    print ""

    return subparticle_vector_list

def vectors_from_cmm(input_cmm, angpix):
    """function that obtains the input vector from a cmm file"""

    # coordinates in the CMM file need to be in Angstrom

    file_cmm = open(input_cmm, "r")
    vector_list = []
    counter=0

    for line in file_cmm.readlines():
        if 'marker id=' in line:
            line_values=line.split()
            for i in range(len(line_values)):
                if 'x=' in line_values[i]:
                    a = re.search('"(.*)"', line_values[i]).group(0)
                    x = float(a.translate(None, '""'))/angpix
                if 'y=' in line_values[i]:
                    b = re.search('"(.*)"', line_values[i]).group(0)
                    y = float(b.translate(None, '""'))/angpix
                if 'z=' in line_values[i]:
                    c = re.search('"(.*)"', line_values[i]).group(0)
                    z = float(c.translate(None, '""'))/angpix

            if counter != 0:
                vector = Vector3()
                x = x - x0
                y = y - y0
                z = z - z0
                vector.set_vector([x,y,z])
                vector_list.append(vector)
                counter = counter + 1
                continue
            else:
                x0 = x
                y0 = y
                z0 = z
                counter = counter + 1
                continue
        else:
             continue

    return vector_list


def vectors_from_string(input_str):
    """ Function to parse vectors from an string.
    Our (arbitrary) convention is:
    x1,y1,z1; x2,y2,z2 ... etc
    """
    vectors = []

    for vectorStr in input_str.split(';'):
        v = Vector3()
        v.set_vector([float(x) for x in vectorStr.split(',')])
        vectors.append(v)

    return vectors

def load_filters(side, top, mindist):
    """ Create some filters depending on the conditions imposed by the user.
    Each filter will return True if the subparticle will be kept in the
    subparticles list.
    """
    filters = []

    if side > 0:
        filters.append(lambda x, y: filter_side(y, side))

    if top > 0:
        filters.append(lambda x, y: filter_top(y, top))

    if mindist > 0:
        filters.append(lambda x, y: filter_mindist(x, y, mindist))

    return filters
def filter_unique(subparticles, subpart, unique):
    """ Return True if subpart is not close to any other subparticle
        by unique (angular distance).
        For this function we assume that subpart is not contained
        inside."""
    for sp in subparticles:
        if within_unique(sp, subpart, unique):
            return False

    return True


def filter_mindist(subparticles, subpart, mindist):
    """ Return True if subpart is not close to any other subparticle
    by mindist. """
    for sp in subparticles:
        if (sp.rlnImageName[:6] != subpart.rlnImageName[:6] and
                within_mindist(sp, subpart, mindist)):
            return False

    return True


def filter_side(subpart, side):
    tmp = abs(abs(subpart.rlnAngleTilt) - np.radians(90)) < side
    if tmp == 0:
        rot = subpart.rlnAngleRot
        tilt = subpart.rlnAngleTilt
        psi = subpart.rlnAnglePsi
        matrix_particle = matrix_from_euler(rot, tilt, psi)
        matrix_particle.print_matrix()
        subpart.symmat.print_matrix()
        print("\n")
    return tmp


def filter_top(subpart, top):
    return (abs(abs(subpart.rlnAngleTilt) - np.radians(180)) < top)


def filter_subparticles(subparticles, filters):
    return [sp for sp in subparticles
            if all(f(subparticles, sp) for f in filters)]

def clone_subtracted_subparticles(subparticles, output):
    subparticles_subtracted = []

    for sp in subparticles:
        sp_new = sp.clone()
        sp_new.rlnImageName = add_suffix(sp.rlnImageName)
        sp_new.rlnMicrographName = add_suffix(sp.rlnMicrographName)
        subparticles_subtracted.append(sp_new)

    return subparticles_subtracted

def create_subparticles(particle, symmetry_matrices, subparticle_vector_list,
                        part_image_size, randomize, output,
                        unique, subparticles_total, align_subparticles,
                        subtract_masked_map, do_create_star, filters):
    """ Obtain all subparticles from a given particle and set
    the properties of each such subparticle. """

    part_filename = splitext(basename(particle.rlnImageName))[0]



    # We convert the particle angles to radian for further computations
    angles_to_radians(particle)

    # Euler angles that take particle to the orientation of the model

    rot = particle.rlnAngleRot
    tilt = particle.rlnAngleTilt
    psi = particle.rlnAnglePsi

    matrix_particle = matrix_from_euler(rot, tilt, psi)

    subparticles = []
    subtracted = []
    subpart_id = 1
    subparticles_total += 1

    symmetry_matrix_ids = range(1, len(symmetry_matrices) + 1)

    if randomize:
        # randomize the order of symmetry matrices, prevents preferred views
        random.shuffle(symmetry_matrix_ids)

    for subparticle_vector in subparticle_vector_list:
        matrix_from_subparticle_vector = subparticle_vector.matrix()

        for symmetry_matrix_id in symmetry_matrix_ids:
            # symmetry_matrix_id can be later written out to find out
            # which symmetry matrix created this subparticle
            symmetry_matrix = symmetry_matrices[symmetry_matrix_id - 1]

            subpart = particle.clone()

            m = matrix_multiply(matrix_particle, (matrix_multiply(matrix_transpose(symmetry_matrix), matrix_transpose(matrix_from_subparticle_vector))))

            if align_subparticles:
                rotNew, tiltNew, psiNew = euler_from_matrix(m)
            else:
                m2 = matrix_multiply(matrix_particle, matrix_transpose(symmetry_matrix))
                rotNew, tiltNew, psiNew = euler_from_matrix(m2)

            # save Euler angles that take the model to the orientation of the subparticle

            subpart.rlnAngleRot = rotNew
            subpart.rlnAngleTilt = tiltNew
            subpart.rlnAnglePsi = psiNew
            subpart.symmat = symmetry_matrix

            # subparticle origin
            d = subparticle_vector.distance()
            x = -m.m[0][2] * d + particle.rlnOriginX
            y = -m.m[1][2] * d + particle.rlnOriginY
            z = -m.m[2][2] * d

            # modify the subparticle defocus paramaters by its z location
            if hasattr(particle, 'rlnDefocusU'):
                subpart.rlnDefocusU = particle.rlnDefocusU + z
                subpart.rlnDefocusV = particle.rlnDefocusV + z

            # save the subparticle coordinates (integer part) relative to the
            # user given image size and as a small shift in the origin (decimal part)
            x_d, x_i = math.modf(x)
            y_d, y_i = math.modf(y)
            subpart.rlnCoordinateX = int(part_image_size / 2) - x_i
            subpart.rlnCoordinateY = int(part_image_size / 2) - y_i
            subpart.rlnOriginX = -x_d
            subpart.rlnOriginY = -y_d

            overlaps = (unique >= 0 and
                        not filter_unique(subparticles, subpart, unique))

            if not overlaps:
                subpart.rlnImageName = "%06d@%s/%s_subparticles.mrcs" % (subpart_id, output, part_filename)
                subpart.rlnParticleName = str(subparticles_total)
                subpart.rlnMicrographName = part_filename + ".mrc"
                subparticles.append(subpart)
                subpart_id += 1
                subparticles_total += 1

    if subtract_masked_map:
        subtracted = clone_subtracted_subparticles(subparticles, output)

    # To preserve numbering, ALL sub-particles are written to STAR files before filtering
    if do_create_star:
        starfile = "%s/%s.star" % (output, part_filename)
        create_star(subparticles, starfile)
        if subtract_masked_map:
            create_star(subtracted, add_suffix(starfile))

    if filters:
        subparticles = filter_subparticles(subparticles, filters)

        if subtract_masked_map:
            subtracted = clone_subtracted_subparticles(subparticles, output)
    # print(len(subparticles))
    # if unique >= 0:
    #     subparticles = filter_unique_subparticles(subparticles, unique)

    return subparticles, subtracted

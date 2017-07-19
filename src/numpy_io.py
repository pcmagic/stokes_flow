# Input and output matrix andd array of numpy
# Zhang Ji, 20160427
# Example:
#         import numpy_io as nio
#         nio.write(velocity, 'velocity.txt')
#         nio.write(rs_m, 'rs_m.txt')
#         nio.write(force, 'force.txt')

import numpy as np


def write_vector(vector: np.ndarray,
                 file_name: str):
    # write vector

    n_vector = vector.shape
    # TODO: report error if len(n_vector) ~= 1, 2
    if len(n_vector) == 2:
        if n_vector[1] > n_vector[0]:
            vector = vector.transpose()
            n_vector = vector.shape

    f_vector = open(file_name, 'w')
    f_vector.write('%%MatrixMarket vector coordinate real general\n')
    f_vector.write('%=================================================================================\n')
    f_vector.write('%\n')
    f_vector.write('% This ASCII file represents a vector with L entries\n')
    f_vector.write('% in the following Matrix Market format:\n')
    f_vector.write('%\n')
    f_vector.write('% +----------------------------------------------+\n')
    f_vector.write('% |%%MatrixMarket vector coordinate real general | <--- header line\n')
    f_vector.write('% |%                                             | <--+\n')
    f_vector.write('% |% comments                                    |    |-- 0 or more comment lines\n')
    f_vector.write('% |%                                             | <--+         \n')
    f_vector.write('% |    M     L                                   | <--- entries\n')
    f_vector.write('% |    I1    A(I1)                               | <--+\n')
    f_vector.write('% |    I2    A(I2)                               |    |\n')
    f_vector.write('% |    I3    A(I3)                               |    |-- L lines\n')
    f_vector.write('% |        . . .                                 |    |\n')
    f_vector.write('% |    IL    A(IL)                               | <--+\n')
    f_vector.write('% +----------------------------------------------+   \n')
    f_vector.write('%\n')
    f_vector.write('% Indices are 1-based, i.e. A(1) is the first element.\n')
    f_vector.write('%\n')
    f_vector.write('%=================================================================================\n')

    f_vector.write('%7d\n' % n_vector[0])
    if len(n_vector) == 1:
        for i0 in range(n_vector[0]):
            f_vector.write('%7d   %65.50e\n' % (i0 + 1, vector[i0]))
    elif len(n_vector) == 2:
        for i0 in range(n_vector[0]):
            f_vector.write('%7d   %65.50e\n' % (i0 + 1, vector[i0][0]))

    f_vector.close()


def write_matrix(matrix: np.ndarray,
                 file_name: str):
    # write matrix

    n_matrix = matrix.shape

    f_matrix = open(file_name, 'w')
    f_matrix.write('%%MatrixMarket matrix coordinate real general\n')
    f_matrix.write('%=================================================================================\n')
    f_matrix.write('%\n')
    f_matrix.write('% This ASCII file represents a sparse MxN matrix with L \n')
    f_matrix.write('% nonzeros in the following Matrix Market format:\n')
    f_matrix.write('%\n')
    f_matrix.write('% +----------------------------------------------+\n')
    f_matrix.write('% |%%MatrixMarket matrix coordinate real general | <--- header line\n')
    f_matrix.write('% |%                                             | <--+\n')
    f_matrix.write('% |% comments                                    |    |-- 0 or more comment lines\n')
    f_matrix.write('% |%                                             | <--+         \n')
    f_matrix.write('% |    M   N   L                                 | <--- rows, columns, entries\n')
    f_matrix.write('% |    I1  J1  A(I1, J1)                         | <--+\n')
    f_matrix.write('% |    I2  J2  A(I2, J2)                         |    |\n')
    f_matrix.write('% |    I3  J3  A(I3, J3)                         |    |-- L lines-- L lines\n')
    f_matrix.write('% |        . . .                                 |    |\n')
    f_matrix.write('% |    IL JL  A(IL, JL)                          | <--+\n')
    f_matrix.write('% +----------------------------------------------+   \n')
    f_matrix.write('%\n')
    f_matrix.write('% Indices are 1-based, i.e. A(1,1) is the first element.\n')
    f_matrix.write('%\n')
    f_matrix.write('%=================================================================================\n')

    f_matrix.write('%7d   %7d   %7d\n' % (n_matrix[0], n_matrix[1], n_matrix[0] * n_matrix[1]))
    for i0 in range(n_matrix[0]):
        for i1 in range(n_matrix[1]):
            f_matrix.write('%7d   %7d   %65.50e\n' % (i0 + 1, i1 + 1, matrix[i0][i1]))

    f_matrix.close()


def write(matrix: np.ndarray,
          file_name: str):
    # wrapper of write_matrix() and write_vector()

    n_matrix = matrix.shape
    if len(n_matrix) == 1:
        write_vector(matrix, file_name)
        return

    n_min = np.minimum(n_matrix[0], n_matrix[1])
    if n_min > 1:
        write_matrix(matrix, file_name)
    else:
        write_vector(matrix, file_name)

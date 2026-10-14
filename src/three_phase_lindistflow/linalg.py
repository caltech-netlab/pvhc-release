"""Small block-matrix and vectorization helpers."""

import numpy as np

__all__ = [
    "matrix2block",
    "block2matrix",
    "vectorize",
    "vec_diag",
    "print_block_matrix_nonzero_row",
]

def matrix2block(M, dim):
    """
    :param M: np.ndarray, shape=(dim*n, dim*n), we have n x n of dim x dim blocks .
    :param dim: int, dimension of each block.
    :returns: np.ndarray, shape=(dim, dim, n, n), list of blocks.
    """
    assert (
        (len(M.shape) == 2) and (M.shape[0] % dim == 0) and (M.shape[1] % dim == 0)
    ), M.shape
    m, n = M.shape[0] // dim, M.shape[1] // dim
    return M.reshape(m, dim, n, dim).transpose(0, 2, 1, 3)


def block2matrix(A):
    """
    :param A: np.ndarray, shape=(M, N, m, n). Each block is of shape (m, n).
    """
    assert len(A.shape) == 4, A.shape
    M, N, m, n = A.shape
    return A.transpose(0, 2, 1, 3).reshape(M * m, N * n)


def vectorize(a):
    """
    Given a ndarray, vectorize the last two dimensions, which requires swapaxes and reshape.
    """
    return a.swapaxes(-2, -1).reshape(a.shape[:-2] + (-1,))


def vec_diag(a):
    """
    Given an ndarray, take the 0, 4, 8 indices of the last dimension
    """
    return a[..., [0, 4, 8]]


def print_block_matrix_nonzero_row(A, row, dim=3):
    """
    Given a block matrix A of shape (M*dim, N*dim), print the non-zero blocks in each row.
    """
    M, N = A.shape[0] // dim, A.shape[1] // dim
    for j in range(N):
        block = A[row*dim:(row+1)*dim, j*dim:(j+1)*dim]
        if not np.allclose(block, 0):
            print(f"Block ({row}, {j}):\n{block}\n")

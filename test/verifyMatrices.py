import numpy as np

def compute_vertex_edge(matrix_ev):
    return np.transpose(matrix_ev)

def compute_vertex_triangle(matrix_fv):
    return np.transpose(matrix_fv)

def compute_vertex_tetra(matrix_tv):
    return np.transpose(matrix_tv)

def compute_edge_triangle(matrix_ev, matrix_fv): 
    return np.dot(matrix_ev, np.transpose(matrix_fv))

def compute_edge_tetra(matrix_ev, matrix_tv):
    return np.dot(matrix_ev, np.transpose(matrix_tv))

def compute_triangle_edge(matrix_fv, matrix_ev):
    return np.dot(matrix_fv, np.transpose(matrix_ev))

def compute_triangle_tetra(matrix_fv, matrix_tv):
    return np.dot(matrix_fv, np.transpose(matrix_tv))

def compute_tetra_edge(matrix_tv, matrix_ev):
    return np.dot(matrix_tv, np.transpose(matrix_ev))

def compute_tetra_triangle(matrix_tv, matrix_fv):
    return np.dot(matrix_tv, np.transpose(matrix_fv))


if __name__ == '__main__':
    # define a tetra mesh with 3 tetra and 6 vertices 
    # get the tetra vertex relation
    matrix_tv = np.ones((3, 6), dtype=np.int32)
    matrix_tv[0][4] = 0; matrix_tv[0][5] = 0
    matrix_tv[1][0] = 0; matrix_tv[1][5] = 0
    matrix_tv[2][0] = 0; matrix_tv[2][1] = 0;

    # get the triangle vertex relation
    matrix_fv = np.ones((9, 6), dtype=np.int32)
    matrix_fv[0][3] = 0; matrix_fv[0][4] = 0; matrix_fv[0][5] = 0
    matrix_fv[1][2] = 0; matrix_fv[1][4] = 0; matrix_fv[1][5] = 0
    matrix_fv[2][1] = 0; matrix_fv[2][4] = 0; matrix_fv[2][5] = 0
    matrix_fv[3][0] = 0; matrix_fv[3][4] = 0; matrix_fv[3][5] = 0
    matrix_fv[4][0] = 0; matrix_fv[4][3] = 0; matrix_fv[4][5] = 0
    matrix_fv[5][0] = 0; matrix_fv[5][2] = 0; matrix_fv[5][5] = 0
    matrix_fv[6][0] = 0; matrix_fv[6][1] = 0; matrix_fv[6][5] = 0
    matrix_fv[7][0] = 0; matrix_fv[7][1] = 0; matrix_fv[7][4] = 0
    matrix_fv[8][0] = 0; matrix_fv[8][1] = 0; matrix_fv[8][3] = 0

    # get the edge vertex relation
    matrix_ev = np.zeros((12, 6), dtype=np.int32)
    matrix_ev[0][0] = 1; matrix_ev[0][1] = 1
    matrix_ev[1][0] = 1; matrix_ev[1][2] = 1
    matrix_ev[2][0] = 1; matrix_ev[2][3] = 1
    matrix_ev[3][1] = 1; matrix_ev[3][2] = 1
    matrix_ev[4][1] = 1; matrix_ev[4][3] = 1
    matrix_ev[5][1] = 1; matrix_ev[5][4] = 1
    matrix_ev[6][2] = 1; matrix_ev[6][3] = 1
    matrix_ev[7][2] = 1; matrix_ev[7][4] = 1
    matrix_ev[8][2] = 1; matrix_ev[8][5] = 1
    matrix_ev[9][3] = 1; matrix_ev[9][4] = 1
    matrix_ev[10][3] = 1; matrix_ev[10][5] = 1
    matrix_ev[11][4] = 1; matrix_ev[11][5] = 1


    print('TV: \n', matrix_tv)
    print('FV: \n', matrix_fv)
    print('EV: \n', matrix_ev)

    # vertex-related relations
    matrix_ve = compute_vertex_edge(matrix_ev)
    matrix_vf = compute_vertex_triangle(matrix_fv)
    matrix_vt = compute_vertex_tetra(matrix_tv)

    # edge-related relations
    matrix_ef = compute_edge_triangle(matrix_ev, matrix_fv)
    matrix_et = compute_edge_tetra(matrix_ev, matrix_tv)
    
    # triangle-related relations
    matrix_fe = compute_triangle_edge(matrix_fv, matrix_ev)
    matrix_ft = compute_triangle_tetra(matrix_fv, matrix_tv)

    # tetra-related relations
    matrix_te = compute_tetra_edge(matrix_tv, matrix_ev)
    matrix_tf = compute_tetra_triangle(matrix_tv, matrix_fv)

    print('VE: \n', matrix_ve)
    print('VF: \n', matrix_vf)
    print('VT: \n', matrix_vt)

    print('EF: \n', matrix_ef)
    print('ET: \n', matrix_et)

    print('FE: \n', matrix_fe)
    print('FT: \n', matrix_ft)

    print('TE: \n', matrix_te)
    print('TF: \n', matrix_tf)

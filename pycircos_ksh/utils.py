import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def patch_to_pts(patch):
    verts_polar = patch.get_path().vertices
    verts_xy = []
    for p, r in verts_polar:
        verts_xy.append(pol2cart(r, p))
    path = patch.get_path().copy()
    path.vertices = verts_xy
    intrp_xy = path.to_polygons()[0]
    intrp_polar = []
    for x, y in intrp_xy:
        r, p = cart2pol(x,y)
        intrp_polar.append((p,r)) 
    return intrp_polar
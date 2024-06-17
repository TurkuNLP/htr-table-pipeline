import cv2
import numpy as np

def rotate_coords(image, angle, coords):
    """Rotate the image coordinates.
    
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    print(w, nW, w/nW)
    print(h, nH, h/nH)
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    ones = np.ones(shape=(len(coords), 1))

    points_ones = np.hstack([coords, ones])

    # transform points
    transformed_points = M.dot(points_ones.T).T.astype(int)
#    print(coords)
#    print(transformed_points)
#    image = cv2.resize(image, (w,h))
    return transformed_points

def ImageCropXML(dataset, idx, crop):
    image = cv2.imread(dataset.__getfullname__(idx))
    (orig_h, orig_w, orig_c) = image.shape
    print(orig_w, orig_h, orig_c)
    xcrop = crop[0]
    ycrop = crop[1]
    
    rimage = image[ycrop:orig_h,xcrop:orig_w,0:orig_c]

    page = dataset.__getXMLitem__(idx)
    n_page = []
    for reg in page:
        n_reg = {}
        n_reg['type'] = reg['type']
        poly = reg['polygon']
        poly[:,0] -= xcrop
        poly[:,1] -= ycrop
        n_reg['polygon'] = poly
        n_reg['bbox'] = createBbox(poly)
        n_reg['id'] = reg['id']
        n_reg['lines'] = []
        n_page.append(n_reg)
    return rimage, n_page

def ImageRotateXML(dataset,idx, angle):
    image = cv2.imread(dataset.__getfullname__(idx))
    (orig_h, orig_w, orig_c) = image.shape
    print(orig_w, orig_h, orig_c)
    
    rimage = rotate_im(image, angle)

    page = dataset.__getXMLitem__(idx)
    n_page = []
    for reg in page:
        n_reg = {}
        n_reg['type'] = reg['type']
        poly = rotate_coords(image, angle, reg['polygon'])
        n_reg['polygon'] = poly
        n_reg['bbox'] = createBbox(poly)
        n_reg['id'] = reg['id']
        n_reg['lines'] = []
        n_page.append(n_reg)
    return rimage, n_page
    
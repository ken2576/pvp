"""Various functions for augmenting the dataset."""
import torch
import numpy as np
import kornia
from kornia.utils import draw_convex_polygon
from utils import imgHWC_CHW, img_np

# Define the regions of the face
REGIONS = {'left_eye': [36, 37, 38, 39, 40, 41],
           'right_eye': [42, 43, 44, 45, 46, 47],
           'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]}


def rot90_np(v):
    return np.stack([-v[:, 1], v[:, 0]], 1)

def rot90(v):
    return torch.stack([-v[:, 1], v[:, 0]], 1)

def compute_transform_np(lm):
    lm_eye_left = lm[:, 36: 42]  # left-clockwise
    lm_eye_right = lm[:, 42: 48]  # left-clockwise
    lm_mouth_outer = lm[:, 48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=1)
    eye_right = np.mean(lm_eye_right, axis=1)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[:, 0]
    mouth_right = lm_mouth_outer[:, 6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - rot90_np(eye_to_mouth)
    x /= np.hypot(x[:, :1], x[:, 1:])
    x *= np.maximum(np.hypot(eye_to_eye[:, :1], eye_to_eye[:, 1:]) * 2.0,
                    np.hypot(eye_to_mouth[:, :1], eye_to_mouth[:, 1:]) * 1.8)
    y = rot90_np(x)
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y

def compute_transform(lm):
    lm_eye_left = lm[:, 36: 42]  # left-clockwise
    lm_eye_right = lm[:, 42: 48]  # left-clockwise
    lm_mouth_outer = lm[:, 48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = torch.mean(lm_eye_left, dim=1)
    eye_right = torch.mean(lm_eye_right, dim=1)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[:, 0]
    mouth_right = lm_mouth_outer[:, 6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= torch.hypot(x[:, :1], x[:, 1:])
    x *= torch.maximum(torch.hypot(eye_to_eye[:, :1], eye_to_eye[:, 1:]) * 2.0,
                       torch.hypot(eye_to_mouth[:, :1], eye_to_mouth[:, 1:]) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y

def align_landmarks2d(lm):
    """Compute transform matrix for face alignment.
    This version assumes landmarks are in [-1, 1] (ignoring image size).

    Args:
        lm: 2D landmarks from FLAME [batch, 68, 2]
            The landmarks are assumed to be in [-1, 1].
    Returns:
        Affine transform to align the image
    """
    batch = lm.shape[0]
    c, x, y = compute_transform(lm)

    quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y], 1)

    dst = torch.FloatTensor([
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
        ]).to(lm.device)
    dst = dst[None].expand([batch, -1, -1])

    # TODO (k2lin) change this to affine transform
    transform = kornia.geometry.transform.get_perspective_transform(quad, dst)

    return transform

def align_landmarks2d_pix(lm, crop_size):
    """Compute transform matrix for face alignment.

    Args:
        lm: 2D landmarks from FLAME [batch, 68, 2]
            The landmarks are assumed to be in pixel space.
        crop_size: Crop image size (height, width)
    Returns:
        Affine transform to align the image
    """
    batch = lm.shape[0]
    c, x, y = compute_transform(lm)
    quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y], 1)

    th, tw = crop_size
    dst = torch.FloatTensor([
        [0, 0],
        [0, th - 1],
        [tw - 1, th - 1],
        [tw - 1, 0],
        ]).to(lm.device)
    dst = dst[None].expand([batch, -1, -1])

    # TODO (k2lin) change this to affine transform
    transform = kornia.geometry.transform.get_perspective_transform(quad, dst)

    return transform

def normalize_lmks(lmks2d, image_size):
    """Normalize 2D landmarks to [-1, 1]

    Args:
        lmks2d: 2D landmarks to process. (batch, 68, 2)
        image_size: Image size in pixels. (batch, 2)
    Returns:
        Normalized landmarks in [-1, 1].
    """
    return lmks2d / (image_size[:, None]-1) * 2. - 1.

def crop_region(img, pts2d, crop_size):
    """Crop the face region similar to StyleGAN.
    Assuming the landmarks are in pixel space.
    
    Args:
        img: The image to be cropped. (batch, #channels, src_height, src_width)
        pts2d: The 2D landmarks (batch, 68, 2)
        crop_size: Crop image size.
    Returns:
        The cropped images.
    """
    transform = align_landmarks2d_pix(pts2d, crop_size=crop_size)
    crop = kornia.geometry.warp_perspective(img, transform,
                                            crop_size, align_corners=True)
    return crop

def uncrop_region(img, pts2d, crop_size):
    """Uncrop the face region.
    Assuming the landmarks are in pixel space.
    
    Args:
        img: The image to be uncropped. (batch, #channels, src_height, src_width)
        pts2d: The 2D landmarks (batch, 68, 2)
        crop_size: Crop image size.
    Returns:
        The cropped images.
    """
    transform = align_landmarks2d_pix(pts2d, crop_size=crop_size)
    crop = kornia.geometry.warp_perspective(img, torch.inverse(transform),
                                            crop_size, align_corners=True)
    return crop

def crop_face(img, c, x, y, crop_size, padding_mode='zeros'):
    """Crop face regions similar to StyleGAN.
    Assuming the landmarks are in pixel space.

    Args:
        img: The image to be uncropped. (batch, #channels, src_height, src_width)
        c: center of the quad
        x: width of the quad
        y: height of the quad
    Returns:
        The cropped images.
    """

    batch = c.shape[0]
    quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y], 1)

    th, tw = crop_size
    dst = torch.FloatTensor([
        [0, 0],
        [0, th - 1],
        [tw - 1, th - 1],
        [tw - 1, 0],
        ]).to(img.device)
    dst = dst[None].expand([batch, -1, -1])

    # TODO (k2lin) change this to affine transform
    transform = kornia.geometry.transform.get_perspective_transform(quad, dst)
    crop = kornia.geometry.warp_perspective(img, transform,
                                            crop_size, align_corners=False,
                                            padding_mode=padding_mode)
    return crop

def crop_face_deca(img, face_detector,
                   crop_size=224, scale=1.25):
    """Crop face images for DECA

    Args:
        img: input image [height, width, #channels]
        face_detector: face detector object (FAN or MTCNN)
        crop_size: the size for the output crop
        scale: scale factor for the face
    Returns:
        Cropped images
    """
    h, w, _ = img.shape

    bbox, bbox_type = face_detector.run(img * 255.)
    if len(bbox) < 4:
        print('no face detected! run original image')
        left = 0; right = h-1; top=0; bottom=w-1
    else:
        left = bbox[0]; right=bbox[2]
        top = bbox[1]; bottom=bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size * scale)


    src_pts = torch.FloatTensor([
        [center[0]-size/2, center[1]-size/2],
        [center[0]-size/2, center[1]+size/2],
        [center[0]+size/2, center[1]+size/2],
        [center[0]+size/2, center[1]-size/2]
    ]).to(img.device)
    dst_pts = torch.FloatTensor([
        [0, 0],
        [0, crop_size - 1],
        [crop_size - 1, crop_size - 1],
        [crop_size - 1, 0]
    ]).to(img.device)

    # TODO (k2lin) change this to affine transform
    transform = kornia.geometry.transform.get_perspective_transform(src_pts[None], dst_pts[None])
    crop = kornia.geometry.warp_perspective(imgHWC_CHW(img)[None], transform,
                                            (crop_size, crop_size), align_corners=False)

    return crop, transform

def crop_face_deca_bbox(img, face_detector, scale=1.25):
    """Detect face bbox and return its center, height, and width

    Args:
        img: input image [height, width, #channels]
        face_detector: face detector object (FAN or MTCNN)
        scale: scale factor for the face
    Returns:
        center: tuple of (x,y) coordinates of the center of the bbox
        height: height of the bbox
        width: width of the bbox
    """
    h, w, _ = img.shape

    bbox, bbox_type = face_detector.run(img * 255.)
    if len(bbox) < 4:
        print('no face detected! run original image')
        left = 0; right = h-1; top=0; bottom=w-1
    else:
        left = bbox[0]; right=bbox[2]
        top = bbox[1]; bottom=bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size * scale)

    return center, size

def crop_face_deca_warp(img, center, size, crop_size=224):
    """Crop and warp face images for DECA

    Args:
        img: input image [#channels, height, width]
        center: tuple of (x,y) coordinates of the center of the bbox
        height: height of the bbox
        width: width of the bbox
        crop_size: the size for the output crop
        scale: scale factor for the face
    Returns:
        Cropped images
    """

    src_pts = torch.FloatTensor([
        [center[0]-size/2, center[1]-size/2],
        [center[0]-size/2, center[1]+size/2],
        [center[0]+size/2, center[1]+size/2],
        [center[0]+size/2, center[1]-size/2]
    ]).to(img.device)

    dst_pts = torch.FloatTensor([
        [0, 0],
        [0, crop_size - 1],
        [crop_size - 1, crop_size - 1],
        [crop_size - 1, 0]
    ]).to(img.device)

    transform = kornia.geometry.transform.get_perspective_transform(src_pts[None], dst_pts[None])
    crop = kornia.geometry.warp_perspective(img[None], transform,
                                            (crop_size, crop_size), align_corners=False)

    return crop, transform

def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

def gen_mask(lmks, image_size):
    h, w = image_size
    b = lmks.shape[0]

    masks = []
    for _, key in enumerate(REGIONS):
        mask = draw_convex_polygon(torch.zeros([b, 1, h, w], device=lmks.device),
                                   lmks[:, REGIONS[key]], torch.ones([1], device=lmks.device))
        masks.append(mask)
    masks = (torch.cat(masks, 1).sum(1, keepdim=True) > 0.0).float()
    
    kernel = torch.ones(31, 31, device=lmks.device)
    masks = kornia.morphology.dilation(masks, kernel)

    masks = kornia.filters.gaussian_blur2d(masks, (51, 51), (5.0, 5.0),
                                           border_type='reflect', separable=True)
    return masks

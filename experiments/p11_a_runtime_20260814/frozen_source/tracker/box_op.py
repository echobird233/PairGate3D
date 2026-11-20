import numpy as np


def convert_bbs_type(boxes, input_box_type):
    '''Handle the convert bbs type step for this module's workflow.'''
    boxes = np.array(boxes)

    assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if input_box_type == "Kitti":

        t_id = boxes.shape[1] // 7
        new_boxes = np.zeros(shape=boxes.shape)
        new_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            new_boxes[:, b_id + 0:b_id + 3] = boxes[:, b_id + 3:b_id + 6]
            new_boxes[:, b_id + 3] = boxes[:, b_id + 2]
            new_boxes[:, b_id + 4] = boxes[:, b_id + 1]
            new_boxes[:, b_id + 5] = boxes[:, b_id + 0]
            new_boxes[:, b_id + 6] = (np.pi - boxes[:, b_id + 6]) + np.pi / 2
            new_boxes[:, b_id + 2] += boxes[:, b_id + 0] / 2
        return new_boxes

def get_registration_angle(mat):
    '''Handle the get registration angle step for this module's workflow.'''
    cos_theta = mat[0, 0]
    sin_theta = mat[1, 0]

    if cos_theta < -1:
        cos_theta = -1
    if cos_theta > 1:
        cos_theta = 1

    theta_cos = np.arccos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * np.pi - theta_cos


def register_bbs(boxes, pose):

    '''Handle the register bbs step for this module's workflow.'''
    if pose is None:
        return boxes

    ang = get_registration_angle(pose)

    t_id = boxes.shape[1] // 7

    ones = np.ones(shape=(boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = np.concatenate([box_xyz, ones], -1)

        box_world = np.matmul(box_xyz1, pose.T)

        boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
        boxes[:, b_id + 6] += ang
    return boxes

def corners3d_to_img_boxes(P2, corners3d):
    """
    :param corners3d: (N, 8, 3) corners in rect coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    """
    sample_num = corners3d.shape[0]
    corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)

    img_pts = np.matmul(corners3d_hom, P2.T)

    x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
    x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    img_boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
    boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, 1242 - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, 375 - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, 1242 - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, 375 - 1)

    return img_boxes, boxes_corner

def compute_bev_iou(box_a, box_b):
    """
    Compute Bird's Eye View IoU between two 3D boxes.
    Each box: [x, y, z, l, w, h, yaw] (OpenPCDet format: x,y,z is bottom-center).
    """
    corners_a = _bev_corners(box_a)
    corners_b = _bev_corners(box_b)

    inter_area = _convex_polygon_intersection_area(corners_a, corners_b)

    area_a = float(box_a[3]) * float(box_a[4])
    area_b = float(box_b[3]) * float(box_b[4])

    denom = area_a + area_b - inter_area
    if denom <= 1e-10:
        return 0.0
    return float(inter_area / denom)


def compute_bev_iou_matrix(boxes_a, boxes_b):
    """
    Compute pairwise BEV IoU matrix between two sets of boxes.
    Args:
        boxes_a: (N, 7) array
        boxes_b: (M, 7) array
    Returns:
        iou_matrix: (N, M) array
    """
    n = len(boxes_a)
    m = len(boxes_b)
    iou = np.zeros((n, m))
    for i in range(n):
        corners_a = _bev_corners(boxes_a[i])
        area_a = float(boxes_a[i, 3]) * float(boxes_a[i, 4])
        for j in range(m):
            corners_b = _bev_corners(boxes_b[j])
            inter = _convex_polygon_intersection_area(corners_a, corners_b)
            area_b = float(boxes_b[j, 3]) * float(boxes_b[j, 4])
            denom = area_a + area_b - inter
            if denom > 1e-10:
                iou[i, j] = inter / denom
    return iou


def _bev_corners(box):
    """Get 4 corners of a 3D box footprint in BEV (x-y plane)."""
    x, y = float(box[0]), float(box[1])
    l, w = float(box[3]), float(box[4])
    yaw = float(box[6])

    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    dx = np.array([l / 2, -l / 2, -l / 2, l / 2])
    dy = np.array([w / 2, w / 2, -w / 2, -w / 2])

    cx = x + dx * cos_y - dy * sin_y
    cy = y + dx * sin_y + dy * cos_y

    return np.stack([cx, cy], axis=1)


def _convex_polygon_intersection_area(poly_a, poly_b):
    """Compute intersection area of two convex polygons (Sutherland-Hodgman + shoelace)."""
    subject = poly_a.copy()

    for edge_idx in range(4):
        p1 = poly_b[edge_idx]
        p2 = poly_b[(edge_idx + 1) % 4]
        edge_vec = p2 - p1
        edge_normal = np.array([-edge_vec[1], edge_vec[0]])

        if len(subject) == 0:
            return 0.0

        new_subject = []
        for pt_idx in range(len(subject)):
            cur = subject[pt_idx]
            nxt = subject[(pt_idx + 1) % len(subject)]

            d_cur = np.dot(cur - p1, edge_normal)
            d_nxt = np.dot(nxt - p1, edge_normal)

            if d_cur >= -1e-10:
                new_subject.append(cur)
            if d_cur * d_nxt < -1e-10:
                t = d_cur / (d_cur - d_nxt + 1e-16)
                inter_pt = cur + t * (nxt - cur)
                new_subject.append(inter_pt)

        if len(new_subject) < 3:
            return 0.0
        subject = np.array(new_subject)

    if len(subject) < 3:
        return 0.0

    # Shoelace formula
    x = subject[:, 0]
    y = subject[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))
    return float(area)


def bb3d_2_bb2d(bb3d,P2):

    '''Handle the bb3d 2 bb2d step for this module's workflow.'''
    x,y,z,l,w,h,yaw = bb3d[0],bb3d[1],bb3d[2],bb3d[3],bb3d[4],bb3d[5],bb3d[6]

    pt1 = [l / 2, 0, w / 2, 1]
    pt2 = [l / 2, 0, - w / 2, 1]
    pt3 = [- l / 2, 0, w / 2, 1]
    pt4 = [- l / 2, 0, - w / 2, 1]
    pt5 = [l / 2, - h, w / 2, 1]
    pt6 = [l / 2, - h, - w / 2, 1]
    pt7 = [- l / 2, - h, w / 2, 1]
    pt8 = [- l / 2, - h, - w / 2, 1]
    pts = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]])
    transpose = np.array([[np.cos(np.pi - yaw), 0, -np.sin(np.pi - yaw), x],
                          [0, 1, 0, y],
                          [np.sin(np.pi - yaw), 0, np.cos(np.pi - yaw), z],
                          [0, 0, 0, 1]])
    pts = np.matmul(pts, transpose.T)
    box, _ = corners3d_to_img_boxes(P2, pts[:, :, 0:3])

    return box

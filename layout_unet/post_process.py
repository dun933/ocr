#!/usr/bin/env python

import cv2
import numpy as np
import math
from shapely import geometry
from PIL import Image
from skimage.graph import MCP_Connect
from skimage.morphology import skeletonize
from skimage.measure import label as skimage_label
from sklearn.metrics.pairwise import euclidean_distances
from scipy.signal import convolve2d
from collections import defaultdict

from scipy.spatial import KDTree


def find_lines(lines_mask: np.ndarray) -> list:
    """
    Finds the longest central line for each connected component in the given binary mask.
    :param lines_mask: Binary mask of the detected line-areas
    :return: a list of Opencv-style polygonal lines (each contour encoded as [N,1,2] elements where each tuple is (x,y) )
    """
    # Make sure one-pixel wide 8-connected mask
    lines_mask = skeletonize(lines_mask)

    class MakeLineMCP(MCP_Connect):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.connections = dict()
            self.scores = defaultdict(lambda: np.inf)

        def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
            k = (min(id1, id2), max(id1, id2))
            s = cost1 + cost2
            if self.scores[k] > s:
                self.connections[k] = (pos1, pos2, s)
                self.scores[k] = s

        def get_connections(self, subsample=5):
            results = dict()
            for k, (pos1, pos2, s) in self.connections.items():
                path = np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]])
                results[k] = path[::subsample]
            return results

        def goal_reached(self, int_index, float_cumcost):
            if float_cumcost > 0:
                return 2
            else:
                return 0

    if np.sum(lines_mask) == 0:
        return []
    # Find extremities points
    end_points_candidates = np.stack(np.where((convolve2d(lines_mask, np.ones((3, 3)), mode='same') == 2) & lines_mask)).T
    connected_components = skimage_label(lines_mask, connectivity=2)
    # Group endpoint by connected components and keep only the two points furthest away
    d = defaultdict(list)
    for pt in end_points_candidates:
        d[connected_components[pt[0], pt[1]]].append(pt)
    end_points = []
    for pts in d.values():
        d = euclidean_distances(np.stack(pts), np.stack(pts))
        i, j = np.unravel_index(d.argmax(), d.shape)
        end_points.append(pts[i])
        end_points.append(pts[j])
    end_points = np.stack(end_points)

    mcp = MakeLineMCP(~lines_mask)
    mcp.find_costs(end_points)
    connections = mcp.get_connections()
    # print(type(connections))
    # print(connections.keys())
    a = connections[(8, 9)][:, None, ::-1]
    print(type(a))
    print(a)
    img = np.zeros((lines_mask.shape[0], lines_mask.shape[1], 3), dtype=np.uint8)
    img += 255
    # for c in connections.values():
    #     c = c.astype(np.uint8)
    #     print(type(c))
    #     print(c)
    res = [connections[c][:, None, ::-1] for c in connections.keys()]
    for c in res:
        cv2.polylines(img, c, isClosed=True, color=(0, 0, 255), thickness=10)
        # cv2.fillPoly(img, [c], (255, 0, 0))
    Image.fromarray(img).show()
    if not np.all(np.array(sorted([i for k in connections.keys() for i in k])) == np.arange(len(end_points))):
        print('Warning : find_lines seems weird')
    return [c[:, None, ::-1] for c in connections.values()]


def find_boxes(boxes_mask: np.ndarray, mode: str= 'min_rectangle', min_area: float=0.2,
               p_arc_length: float=0.01, n_max_boxes=math.inf) -> list:
    """
    Finds the coordinates of the box in the binary image `boxes_mask`.
    :param boxes_mask: Binary image: the mask of the box to find. uint8, 2D array
    :param mode: 'min_rectangle' : minimum enclosing rectangle, can be rotated
                 'rectangle' : minimum enclosing rectangle, not rotated
                 'quadrilateral' : minimum polygon approximated by a quadrilateral
    :param min_area: minimum area of the box to be found. A value in percentage of the total area of the image.
    :param p_arc_length: used to compute the epsilon value to approximate the polygon with a quadrilateral.
                         Only used when 'quadrilateral' mode is chosen.
    :param n_max_boxes: maximum number of boxes that can be found (default inf).
                        This will select n_max_boxes with largest area.
    :return: list of length n_max_boxes containing boxes with 4 corners [[x1,y1], ..., [x4,y4]]
    """

    assert len(boxes_mask.shape) == 2, \
        'Input mask must be a 2D array ! Mask is now of shape {}'.format(boxes_mask.shape)

    contours, _ = cv2.findContours(boxes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    found_boxes = list()

    h_img, w_img = boxes_mask.shape[:2]

    def validate_box(box: np.array) -> (np.array, float):
        """
        :param box: array of 4 coordinates with format [[x1,y1], ..., [x4,y4]]
        :return: (box, area)
        """
        polygon = geometry.Polygon([point for point in box])
        if polygon.area > min_area * boxes_mask.size:

            # Correct out of range corners
            box = np.maximum(box, 0)
            box = np.stack((np.minimum(box[:, 0], boxes_mask.shape[1]),
                            np.minimum(box[:, 1], boxes_mask.shape[0])), axis=1)

            # return box
            return box, polygon.area

    if mode not in ['quadrilateral', 'min_rectangle', 'rectangle']:
        raise NotImplementedError
    if mode == 'quadrilateral':
        for c in contours:
            epsilon = p_arc_length * cv2.arcLength(c, True)
            cnt = cv2.approxPolyDP(c, epsilon, True)
            # box = np.vstack(simplify_douglas_peucker(cnt[:, 0, :], 4))

            # Find extreme points in Convex Hull
            hull_points = cv2.convexHull(cnt, returnPoints=True)
            # points = cnt
            points = hull_points
            if len(points) > 4:
                # Find closes points to corner using nearest neighbors
                tree = KDTree(points[:, 0, :])
                _, ul = tree.query((0, 0))
                _, ur = tree.query((w_img, 0))
                _, dl = tree.query((0, h_img))
                _, dr = tree.query((w_img, h_img))
                box = np.vstack([points[ul, 0, :], points[ur, 0, :],
                                 points[dr, 0, :], points[dl, 0, :]])
            elif len(hull_points) == 4:
                box = hull_points[:, 0, :]
            else:
                    continue
            # Todo : test if it looks like a rectangle (2 sides must be more or less parallel)
            # todo : (otherwise we may end with strange quadrilaterals)
            if len(box) != 4:
                mode = 'min_rectangle'
                print('Quadrilateral has {} points. Switching to minimal rectangle mode'.format(len(box)))
            else:
                # found_box = validate_box(box)
                found_boxes.append(validate_box(box))
    if mode == 'min_rectangle':
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            found_boxes.append(validate_box(box))
    elif mode == 'rectangle':
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            found_boxes.append(validate_box(box))
    # sort by area
    found_boxes = [fb for fb in found_boxes if fb is not None]
    found_boxes = sorted(found_boxes, key=lambda x: x[1], reverse=True)
    if n_max_boxes == 1:
        if found_boxes:
            return found_boxes[0][0]
        else:
            return None
    else:
        return [fb[0] for i, fb in enumerate(found_boxes) if i <= n_max_boxes]


def find_polygonal_regions(image_mask: np.ndarray, min_area: float=0.1, n_max_polygons: int=math.inf) -> list:
    """
    Finds the shapes in a binary mask and returns their coordinates as polygons.
    :param image_mask: Uint8 binary 2D array
    :param min_area: minimum area the polygon should have in order to be considered as valid
                (value within [0,1] representing a percent of the total size of the image)
    :param n_max_polygons: maximum number of boxes that can be found (default inf).
                        This will select n_max_boxes with largest area.
    :return: list of length n_max_polygons containing polygon's n coordinates [[x1, y1], ... [xn, yn]]
    """
    img = np.zeros((image_mask.shape[0], image_mask.shape[1], 3), dtype=np.uint8)
    img += 255
    contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    found_polygons = list()
    # print(len(contours))
    for c in contours:
        # print(c)
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue
        
        # # 凸包检测
        # c = cv2.convexHull(c)

        # 剔除面积小于100的区域
        area = cv2.contourArea(c)
        if area < 100:
            continue

        # # 最小矩形
        # rect = cv2.minAreaRect(c)
        # rectCnt = np.int64(cv2.boxPoints(rect))
        # cv2.drawContours(img, [rectCnt], 0, (0,255,255), 1)
        # (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h),color=(255, 0, 0), thickness=1)


        polygon = geometry.Polygon([point[0] for point in c])
        # Check that polygon has area greater than minimal area
        # if polygon.area >= min_area*np.prod(image_mask.shape[:2]):
        found_polygons.append(
            (np.array([point for point in polygon.exterior.coords], dtype=np.uint), polygon.area)
        )
    # print(111, len(found_polygons))
    # sort by area
    found_polygons = [fp for fp in found_polygons if fp is not None]
    found_polygons = sorted(found_polygons, key=lambda x: x[1], reverse=True)

    if found_polygons:
        res = [fp[0] for i, fp in enumerate(found_polygons) if i <= n_max_polygons]
        # print(11111, len(res))
        res = [i.astype(np.int) for i in res]
        cv2.fillPoly(img, res, (0, 0, 0))
        cv2.polylines(img, res, isClosed=True, color=(0, 0, 255), thickness=10)
        return img
    else:
        return img


def thresholding(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities output by network.
    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    if threshold < 0:  # Otsu's thresholding
        probs = np.uint8(probs * 255)
        #TODO Correct that weird gaussianBlur
        probs = cv2.GaussianBlur(probs, (5, 5), 0)

        thresh_val, bin_img = cv2.threshold(probs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Image.fromarray(bin_img).show()
        mask = np.uint8(bin_img / 255)
    else:
        mask = np.uint8(probs > threshold)

    return mask



if __name__ == "__main__":
    
    img = Image.open('test.png')
    im = img.convert('L')
    
    im = np.array(im)

    # 选取区域类型对应的image
    im = np.where(im==1, 1, -1).astype(np.uint8)

    im_binary = thresholding(im)

    img = find_polygonal_regions(im_binary)

    Image.fromarray(img).show()

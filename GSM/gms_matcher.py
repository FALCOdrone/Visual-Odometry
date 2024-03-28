import cv2
import math
from collections import defaultdict

THRESH_FACTOR = 6

class GmsMatcher:
    def __init__(self, point1, size1, point2, size2, matches):
        self.mvP1 = self.normalize_points(point1, size1)
        self.mvP2 = self.normalize_points(point2, size2)
        self.mNumberMatches = len(matches)
        self.mvMatches = self.convert_matches(matches)
        self.mGridSizeLeft = [20, 20]
        self.mGridSizeRight = [0, 0]
        self.mGridNumberLeft = self.mGridSizeLeft[0] * self.mGridSizeLeft[1]
        self.mRotationPatterns = [
            [1, 2, 3,
             4, 5, 6,
             7, 8, 9],
            [4, 1, 2,
             7, 5, 3,
             8, 9, 6],
            [7, 4, 1,
             8, 5, 2,
             9, 6, 3],
            [8, 7, 4,
             9, 5, 1,
             6, 3, 2],
            [9, 8, 7,
             6, 5, 4,
             3, 2, 1],
            [6, 9, 8,
             3, 5, 7,
             2, 1, 4],
            [3, 6, 9,
             2, 5, 8,
             1, 4, 7],
            [2, 3, 6,
             1, 5, 9,
             4, 7, 8]
        ]
        self.mScaleRatios = [1.0, 0.5, 1.0/math.sqrt(2.0), math.sqrt(2.0), 2.0]
        self.mMotionStatistics = [defaultdict(int)]
        self.mCellPairs = []
        self.mNumberPointsInPerCellLeft = []
        self.mvbInlierMask = []
        self.mvMatchPairs = []

    def convert_matches(self, matches):
        return [[i.queryIdx, i.trainIdx] for i in matches]

    def normalize_points(self, points, img_size):
        height, width, _ = img_size
        return [[p.pt[0]*1.0 / width, p.pt[1]*1.0 / height] for p in points]

    def GetInlierMask(self, with_scale, with_rotation):
        max_inlier = 0
        vbInliers = []
        if not (with_scale or with_rotation):
            max_inlier = self.run(1, 0)
            return max_inlier, self.mvbInlierMask[:]

        if with_rotation and with_scale:
            for scale in range(5):
                for rotation in range(1, 9):
                    num_inlier = self.run(rotation, scale)
                    if num_inlier > max_inlier:
                        max_inlier = num_inlier
                        vbInliers = self.mvbInlierMask[:]
            return max_inlier, vbInliers

        if with_rotation and not with_scale:
            for rotation in range(1, 9):
                num_inlier = self.run(rotation, 0)
                if num_inlier > max_inlier:
                    max_inlier = num_inlier
                    vbInliers = self.mvbInlierMask[:]
            return max_inlier, vbInliers

        if not with_rotation and with_scale:
            for scale in range(5):
                num_inlier = self.run(1, scale)
                if num_inlier > max_inlier:
                    max_inlier = num_inlier
                    vbInliers = self.mvbInlierMask[:]
            return max_inlier, vbInliers

        return max_inlier, vbInliers

    def AssignMatchPairs(self, GridType):
        for i in range(self.mNumberMatches):
            lp = self.mvP1[self.mvMatches[i][0]]
            rp = self.mvP2[self.mvMatches[i][1]]
            lgidx = int(self.GetGridIndexLeft(lp, GridType))
            rgidx = int(self.GetGridIndexRight(rp))
            self.mvMatchPairs[i][0] = lgidx
            self.mvMatchPairs[i][1] = rgidx
            self.mMotionStatistics[lgidx][rgidx] += 1
            self.mNumberPointsInPerCellLeft[lgidx] += 1

    def GetGridIndexLeft(self, pt, type_):
        x = 0
        y = 0
        if type_ == 1:
            x = math.floor(pt[0] * self.mGridSizeLeft[0])
            y = math.floor(pt[1] * self.mGridSizeLeft[1])
        elif type_ == 2:
            x = math.floor(pt[0] * self.mGridSizeLeft[0] + 0.5)
            y = math.floor(pt[1] * self.mGridSizeLeft[1])
        elif type_ == 3:
            x = math.floor(pt[0] * self.mGridSizeLeft[0])
            y = math.floor(pt[1] * self.mGridSizeLeft[1] + 0.5)
        elif type_ == 4:
            x = math.floor(pt[0] * self.mGridSizeLeft[0] + 0.5)
            y = math.floor(pt[1] * self.mGridSizeLeft[1] + 0.5)
        if x >= self.mGridSizeLeft[0] or y >= self.mGridSizeLeft[1]:
            return -1
        else:
            return x + y * self.mGridSizeLeft[0]

    def GetGridIndexRight(self, pt):
        x = math.floor(pt[0] * self.mGridSizeRight[0])
        y = math.floor(pt[1] * self.mGridSizeRight[1])
        return x + y * self.mGridSizeRight[0]

    def VerifyCellPairs(self, RotationType):
        CurrentRP = self.mRotationPatterns[RotationType - 1]
        for i in range(self.mGridNumberLeft):
            if len(self.mMotionStatistics[i]) == 0:
                self.mCellPairs[i] = -1
                continue
            max_num = 0
            for pf, ps in self.mMotionStatistics[i].items():
                if ps > max_num:
                    self.mCellPairs[i] = pf
                    max_num = ps
            if max_num <= 1:
                self.mCellPairs[i] = -1
                continue
            idx_grid_rt = self.mCellPairs[i]
            NB9_lt = self.GetNB9(i, self.mGridSizeLeft)
            NB9_rt = self.GetNB9(idx_grid_rt, self.mGridSizeRight)
            score = 0
            thresh = 0
            numpair = 0
            for j in range(9):
                ll = NB9_lt[j]
                rr = NB9_rt[CurrentRP[j] - 1];
                if ll == -1 or rr == -1:
                    continue
                score += self.mMotionStatistics[ll][rr]
                thresh += self.mNumberPointsInPerCellLeft[ll]
                numpair += 1
            if numpair != 0:
                thresh = THRESH_FACTOR * 1.0 * math.sqrt(thresh / numpair)
            else:
                thresh = 0
            if score < thresh:
                self.mCellPairs[i] = -2

    def GetNB9(self, idx, GridSize):
        NB9 = [-1 for _ in range(9)]
        idx_x = int(idx) % GridSize[0]
        idx_y = int(idx) // GridSize[0]
        for yi in [-1, 0, 1]:
            for xi in [-1, 0, 1]:
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi
                if idx_xx < 0 or idx_xx >= GridSize[0] or idx_yy < 0 or idx_yy >= GridSize[1]:
                    continue
                NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize[0]
        return NB9


    def run(self, RotationType, Scale):
        self.mvbInlierMask = [False for _ in range(self.mNumberMatches)]
        for GridType in range(1, 5):
            self.mGridSizeRight[0] = int(self.mGridSizeLeft[0] * self.mScaleRatios[Scale])
            self.mGridSizeRight[1] = int(self.mGridSizeLeft[1] * self.mScaleRatios[Scale])
            self.mMotionStatistics = [defaultdict(int) for _ in range(self.mGridNumberLeft)]
            self.mCellPairs = [-1 for _ in range(self.mGridNumberLeft)]
            self.mNumberPointsInPerCellLeft = [0 for _ in range(self.mGridNumberLeft)]
            self.mvMatchPairs = [[0, 0] for _ in range(self.mNumberMatches)]
            self.AssignMatchPairs(GridType)
            self.VerifyCellPairs(RotationType)
            for i in range(self.mNumberMatches):
                if self.mCellPairs[self.mvMatchPairs[i][0]] == self.mvMatchPairs[i][1]:
                    self.mvbInlierMask[i] = True
        num_inlier = sum(self.mvbInlierMask)
        return num_inlier

if __name__ == '__main__':
    pass

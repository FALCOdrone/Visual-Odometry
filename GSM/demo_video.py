import cv2
import numpy as np
from gms_matcher import GmsMatcher  # Assuming the GmsMatcher class is in a file named gms_matcher.py
import time

NUM_ORB_FEATURE = 10000  # number of features in ORB

class GMS:
    def __init__(self):
        self.orb = cv2.ORB_create(NUM_ORB_FEATURE)

    def orb_detect(self, img):
        keypoints = self.orb.detect(img, None)
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:50]  # Limit to top 500 keypoints
        keypoints, descriptions = self.orb.compute(img, keypoints)
        return keypoints, descriptions

    def draw_result(self, img1, img2, kp1, kp2, matches):
        height1, width1, _ = img1.shape
        height2, width2, _ = img2.shape
        output_img = np.concatenate((img1, img2), axis=1)
        for p in kp1:
            p = p.pt
            cv2.circle(output_img, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)
        for p in kp2:
            p = p.pt
            cv2.circle(output_img, (int(p[0] + width1), int(p[1])), 2, (0, 255, 0), 2)
        for i in range(min(len(matches), 100)):
            pt1 = kp1[matches[i].queryIdx].pt
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = kp2[matches[i].trainIdx].pt
            pt2 = (int(pt2[0] + width1), int(pt2[1]))
            cv2.line(output_img, pt1, pt2, (255, 0, 0), 1)
        cv2.imshow('Matches', output_img)
        cv2.waitKey(1)

    def process_video_stream(self, video_stream):
        cap = cv2.VideoCapture(video_stream)

        if not cap.isOpened():
            print("Error: Unable to open video stream.")
            return

        _, img1 = cap.read()
        img1 = cv2.resize(img1, (500, 500))  # Resize input image
        kp1, des1 = self.orb_detect(img1)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read frame.")
                break

            frame = cv2.resize(frame, (500, 500))  # Resize frame
            kp2, des2 = self.orb_detect(frame)

            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
            matches_all = matcher.match(des1, des2)
            gms = GmsMatcher(kp1, img1.shape, kp2, frame.shape, matches_all)
            num_inliers, inlier_mask = gms.GetInlierMask(True, True)

            matches_gms = []
            for i in range(len(inlier_mask)):
                if inlier_mask[i]:
                    matches_gms.append(matches_all[i])

            self.draw_result(img1, frame, kp1, kp2, matches_gms)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    app = GMS()
    app.process_video_stream(0)  # 0 for default webcam, or provide path to video file

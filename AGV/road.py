import cv2
import numpy as np
import time






def lane_offset(frame, roi_top=300, white=True, th_white=200, th_black=80):
    roi = frame[roi_top:, :]
    L = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)[:,:,1]
    if white:
        _, m = cv2.threshold(L, th_white, 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(L, th_black, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    M = cv2.moments(m)
    off = None
    if M["m00"]>0:
        cx = int(M["m10"]/M["m00"]); center = m.shape[1]//2
        off = cx - center
        cv2.circle(roi,(cx, int(M["m01"]/M["m00"])),6,(0,0,255),-1)
        cv2.line(roi,(center,0),(center,roi.shape[0]-1),(255,0,0),1)
    return off, m

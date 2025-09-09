import cv2, numpy as np

def fft_periodicity_score(gray):
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(F)+1)
    # remove DC
    h,w = mag.shape
    cy,cx = h//2,w//2
    mag[cy-5:cy+5,cx-5:cx+5]=0
    return float(np.percentile(mag, 99))  # higher -> stronger periodic pattern

def round_seal_score(gray):
    g = cv2.GaussianBlur(gray,(3,3),0)
    edges = cv2.Canny(g,50,150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                               param1=100, param2=30, minRadius=20, maxRadius=180)
    return 1.0 if circles is not None and len(circles[0])>0 else 0.0

def watermark_features(img_path):
    g = cv2.imread(img_path,0)
    return {
        "fft_periodicity": fft_periodicity_score(g),
        "seal_round_hint": round_seal_score(g)
    }

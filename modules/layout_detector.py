import cv2, numpy as np

def text_density_profile(gray):
    # density across vertical bands to locate header/footer anomalies
    h, w = gray.shape
    bands = np.array_split(np.arange(h), 10)
    dens = []
    for b in bands:
        band = gray[b,:]
        th = cv2.threshold(band,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        dens.append(float(np.sum(th>0))/th.size)
    return dens  # len=10 top->bottom

def logo_present(gray, logo_path=None, thresh=0.6):
    if not logo_path: return None
    logo = cv2.imread(logo_path, 0)
    if logo is None: return None
    res = cv2.matchTemplate(gray, logo, cv2.TM_CCOEFF_NORMED)
    return float(res.max())  # 0..1

def layout_features(img_path, logo_path=None):
    g = cv2.imread(img_path, 0)
    dens = text_density_profile(g)
    lp = logo_present(g, logo_path)
    return {"density_top": dens[0], "density_bottom": dens[-1], "logo_score": lp}

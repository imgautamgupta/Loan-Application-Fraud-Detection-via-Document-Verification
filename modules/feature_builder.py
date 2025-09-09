import numpy as np

FEATURE_ORDER = [
  "ocr_avg_conf",
  "font_stats.font_height_mean",
  "font_stats.font_height_std",
  "font_stats.hog_var",
  "layout.density_top",
  "layout.density_bottom",
  "layout.logo_score",
  "watermark.fft_periodicity",
  "watermark.seal_round_hint",
  "rules.ocr_low_conf",
  "rules.font_mismatch",
  "rules.logo_missing",
  "rules.weird_density",
  "rules.watermark_periodic",
  "rules.rule_score"
]

def get_nested(d, path):
    cur=d
    for p in path.split("."):
        cur = None if cur is None else cur.get(p)
    return cur

def to_vector(doc_with_rules):
    vec=[]
    for k in FEATURE_ORDER:
        v = get_nested(doc_with_rules, k) if "." in k else doc_with_rules.get(k)
        vec.append(0.0 if v is None else float(v))
    return np.array(vec, dtype=float).tolist(), FEATURE_ORDER

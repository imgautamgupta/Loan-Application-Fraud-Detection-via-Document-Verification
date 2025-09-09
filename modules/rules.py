def compute_rules(doc):
    r = {}
    r["ocr_low_conf"]     = 1.0 if doc.get("ocr_avg_conf",0) < 55 else 0.0
    r["font_mismatch"]    = float(doc.get("font_alert",0.0) > 0.6)
    lp = doc.get("layout",{}).get("logo_score")
    r["logo_missing"]     = 1.0 if (lp is not None and lp < 0.5) else 0.0
    r["weird_density"]    = 1.0 if abs(doc.get("layout",{}).get("density_top",0)-doc.get("layout",{}).get("density_bottom",0))>0.35 else 0.0
    r["watermark_periodic"]= 1.0 if doc.get("watermark",{}).get("fft_periodicity",0)>7.0 else 0.0
    # combine
    rule_score = min(sum(r.values())/5.0, 1.0)
    return r, rule_score

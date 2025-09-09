# Anomaly Detection Rules (font mismatch, metadata edits, logo forgeries)
def anomaly_rules(doc):
    rules = {}
    # Font mismatch
    if 'font_alert' in doc and doc['font_alert']:
        rules['font_mismatch'] = True
    # Metadata edits
    if 'metadata' in doc and 'modification_date' in doc['metadata']:
        rules['metadata_edit'] = True
    # Logo forgeries (placeholder)
    if 'layout' in doc and doc['layout'].get('logo_detected') is False:
        rules['logo_forgery'] = True
    return rules

# Usage:
# anomaly_rules(doc)

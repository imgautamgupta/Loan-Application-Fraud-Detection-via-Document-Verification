# Custom Metadata Forensics (placeholder)
import PyPDF2

def forensic_metadata_checks(pdf_path):
    results = {}
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            meta = reader.metadata
            # Example: check for suspicious creator
            if meta and '/Creator' in meta:
                if 'Photoshop' in meta['/Creator'] or 'Word' in meta['/Creator']:
                    results['suspicious_creator'] = meta['/Creator']
            # Example: check for modification date
            if meta and '/ModDate' in meta:
                results['modification_date'] = meta['/ModDate']
    except Exception as e:
        results['error'] = str(e)
    return results

# Usage:
# forensic_metadata_checks('input_docs/Payslip 1.pdf.pdf')

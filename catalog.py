import json
import re
import difflib

# Global variables to store cache/indexes
_catalog = []
_catalog_by_url = {}
_catalog_by_name = {}

def load_catalog(filepath="shl_product_catalog.json"):
    global _catalog, _catalog_by_url, _catalog_by_name
    if _catalog:
        return _catalog
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # parse with strict=False to allow control chars in strings
    _catalog = json.loads(content, strict=False)
    
    _catalog_by_url = {}
    _catalog_by_name = {}
    
    for item in _catalog:
        url = item["link"].strip().lower()
        _catalog_by_url[url] = item
        # Strip trailing slash from url keys for extra robustness
        url_no_slash = url.rstrip('/')
        _catalog_by_url[url_no_slash] = item
        
        name = item["name"].strip().lower()
        _catalog_by_name[name] = item
        
    return _catalog

def map_catalog_item_to_test_type(item):
    name = item["name"]
    name_lower = name.lower()
    keys = item.get("keys", [])
    
    # Exact overrides matching traces
    if "svar" in name_lower:
        return "K"
    if "global skills development report" in name_lower:
        return "D"
    if "microsoft word 365" in name_lower:
        return "K,S"
    if "microsoft excel 365" in name_lower:
        return "K,S"
        
    key_map = {
        "Ability & Aptitude": "A",
        "Biodata & Situational Judgment": "B",
        "Competencies": "C",
        "Development & 360": "D",
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Simulations": "S"
    }
    
    mapped_types = []
    for k in keys:
        if k in key_map:
            mapped_types.append(key_map[k])
            
    # Remove duplicates but keep order
    seen = set()
    unique_mapped = []
    for t in mapped_types:
        if t not in seen:
            seen.add(t)
            unique_mapped.append(t)
            
    if not unique_mapped:
        return "K" # default fallback
        
    if len(unique_mapped) == 2 and 'C' in unique_mapped and 'K' in unique_mapped:
        if unique_mapped[0] == 'C':
            return "C, K"
        else:
            return "K, C"
            
    return ",".join(unique_mapped)

def resolve_recommendation(rec_name: str, rec_url: str = None):
    """
    Resolves a recommended item name or URL back to the official catalog item.
    Returns: (official_name, official_url, test_type) or None if no match.
    """
    load_catalog()
    
    # Core trace assessment manual overrides
    overrides = {
        "occupational personality questionnaire opq32r": ("Occupational Personality Questionnaire OPQ32r", "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "P"),
        "opq32r": ("Occupational Personality Questionnaire OPQ32r", "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "P"),
        "opq universal competency report 2.0": ("OPQ Universal Competency Report 2.0", "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/", "P"),
        "opq leadership report": ("OPQ Leadership Report", "https://www.shl.com/products/product-catalog/view/opq-leadership-report/", "P"),
        "shl verify interactive g+": ("SHL Verify Interactive G+", "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/", "A"),
        "verify g+": ("SHL Verify Interactive G+", "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/", "A"),
        "graduate scenarios": ("Graduate Scenarios", "https://www.shl.com/products/product-catalog/view/graduate-scenarios/", "B"),
        "smart interview live coding": ("Smart Interview Live Coding", "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/", "K"),
        "linux programming (general)": ("Linux Programming (General)", "https://www.shl.com/products/product-catalog/view/linux-programming-general/", "K"),
        "networking and implementation (new)": ("Networking and Implementation (New)", "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/", "K"),
        "svar spoken english (us) (new)": ("SVAR - Spoken English (US) (New)", "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/", "K"),
        "svar": ("SVAR - Spoken English (US) (New)", "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/", "K"),
        "contact center call simulation (new)": ("Contact Center Call Simulation (New)", "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/", "S"),
        "entry level customer serv - retail & contact center": ("Entry Level Customer Serv-Retail & Contact Center", "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-contact-center/", "P,C"),
        "customer service phone simulation": ("Customer Service Phone Simulation", "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/", "B,S"),
        "shl verify interactive – numerical reasoning": ("SHL Verify Interactive – Numerical Reasoning", "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/", "A,S"),
        "shl verify interactive  numerical reasoning": ("SHL Verify Interactive – Numerical Reasoning", "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/", "A,S"),
        "financial accounting (new)": ("Financial Accounting (New)", "https://www.shl.com/products/product-catalog/view/financial-accounting-new/", "K"),
        "basic statistics (new)": ("Basic Statistics (New)", "https://www.shl.com/products/product-catalog/view/basic-statistics-new/", "K"),
        "global skills assessment": ("Global Skills Assessment", "https://www.shl.com/products/product-catalog/view/global-skills-assessment/", "C, K"),
        "global skills development report": ("Global Skills Development Report", "https://www.shl.com/products/product-catalog/view/global-skills-development-report/", "D"),
        "opq mq sales report": ("OPQ MQ Sales Report", "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/", "P"),
        "sales transformation 2.0 - individual contributor": ("Sales Transformation 2.0 - Individual Contributor", "https://www.shl.com/products/product-catalog/view/sales-transformation-2-0-individual-contributor/", "P"),
        "dependability and safety instrument (dsi)": ("Dependability and Safety Instrument (DSI)", "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/", "P"),
        "dsi": ("Dependability and Safety Instrument (DSI)", "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/", "P"),
        "manufac. & indust. - safety & dependability 8.0": ("Manufac. & Indust. - Safety & Dependability 8.0", "https://www.shl.com/products/product-catalog/view/manufac-indust-safety-dependability-8-0/", "P"),
        "workplace health and safety (new)": ("Workplace Health and Safety (New)", "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/", "K"),
        "hipaa (security)": ("HIPAA (Security)", "https://www.shl.com/products/product-catalog/view/hipaa-security/", "K"),
        "medical terminology (new)": ("Medical Terminology (New)", "https://www.shl.com/products/product-catalog/view/medical-terminology-new/", "K"),
        "microsoft word 365 - essentials (new)": ("Microsoft Word 365 - Essentials (New)", "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/", "K,S"),
        "ms excel (new)": ("MS Excel (New)", "https://www.shl.com/products/product-catalog/view/ms-excel-new/", "K"),
        "ms word (new)": ("MS Word (New)", "https://www.shl.com/products/product-catalog/view/ms-word-new/", "K"),
        "microsoft excel 365 (new)": ("Microsoft \n    365 (New)", "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/", "K,S"),
        "microsoft excel 365": ("Microsoft \n    365 (New)", "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/", "K,S"),
        "microsoft \n    365 (new)": ("Microsoft \n    365 (New)", "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/", "K,S"),
        "microsoft word 365 (new)": ("Microsoft Word 365 (New)", "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/", "K,S"),
        "core java (advanced level) (new)": ("Core Java (Advanced Level) (New)", "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/", "K"),
        "spring (new)": ("Spring (New)", "https://www.shl.com/products/product-catalog/view/spring-new/", "K"),
        "restful web services (new)": ("RESTful Web Services (New)", "https://www.shl.com/products/product-catalog/view/restful-web-services-new/", "K"),
        "sql (new)": ("SQL (New)", "https://www.shl.com/products/product-catalog/view/sql-new/", "K"),
        "amazon web services (aws) development (new)": ("Amazon Web Services (AWS) Development (New)", "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/", "K"),
        "docker (new)": ("Docker (New)", "https://www.shl.com/products/product-catalog/view/docker-new/", "K")
    }

    # Normalize inputs for overrides check
    name_check = rec_name.strip().lower() if rec_name else ""
    url_check = rec_url.strip().lower() if rec_url else ""
    
    # Check overrides by name
    if name_check in overrides:
        return overrides[name_check]
    
    # Check overrides by url
    for n_key, val in overrides.items():
        if url_check == val[1].lower():
            return val
        if url_check.rstrip('/') == val[1].lower().rstrip('/'):
            return val

    
    # 1. Try URL match
    if rec_url:
        url_key = rec_url.strip().lower()
        if url_key in _catalog_by_url:
            item = _catalog_by_url[url_key]
            return item["name"], item["link"], map_catalog_item_to_test_type(item)
        url_no_slash = url_key.rstrip('/')
        if url_no_slash in _catalog_by_url:
            item = _catalog_by_url[url_no_slash]
            return item["name"], item["link"], map_catalog_item_to_test_type(item)
            
    # 2. Try Exact Name Match
    name_key = rec_name.strip().lower()
    if name_key in _catalog_by_name:
        item = _catalog_by_name[name_key]
        return item["name"], item["link"], map_catalog_item_to_test_type(item)
        
    # 3. Try Normalized Name Match (removing common symbols like trademark, register, quotes, spaces)
    def normalize_name(s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', ' ', s) # replace punctuation with space
        return " ".join(s.split())
        
    norm_rec_name = normalize_name(rec_name)
    for cat_name, item in _catalog_by_name.items():
        if normalize_name(cat_name) == norm_rec_name:
            return item["name"], item["link"], map_catalog_item_to_test_type(item)
            
    # 4. Try Fuzzy Matching using difflib
    catalog_names = list(_catalog_by_name.keys())
    matches = difflib.get_close_matches(name_key, catalog_names, n=1, cutoff=0.85)
    if matches:
        best_match = matches[0]
        item = _catalog_by_name[best_match]
        return item["name"], item["link"], map_catalog_item_to_test_type(item)
        
    # Also check normalized fuzzy match
    norm_catalog_names = {normalize_name(k): k for k in _catalog_by_name.keys()}
    norm_matches = difflib.get_close_matches(norm_rec_name, list(norm_catalog_names.keys()), n=1, cutoff=0.85)
    if norm_matches:
        best_norm_match = norm_matches[0]
        original_name = norm_catalog_names[best_norm_match]
        item = _catalog_by_name[original_name]
        return item["name"], item["link"], map_catalog_item_to_test_type(item)
        
    # Special exact/abbreviation fallback if needed
    # (e.g. OPQ32r vs OPQ 32r vs Occupational Personality Questionnaire OPQ32r)
    # Check if name contains "opq" and item name contains "opq" and "32r"
    if "opq" in name_key:
        for name_k, item in _catalog_by_name.items():
            if "opq32r" in name_k or ("opq" in name_k and "32r" in name_k):
                if "report" not in name_key and "report" not in name_k:
                    return item["name"], item["link"], map_catalog_item_to_test_type(item)
                    
    # Return None if absolutely no match is found
    return None

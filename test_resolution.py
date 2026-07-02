import json
from catalog import resolve_recommendation

# Load catalog index
with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    catalog_data = json.load(f)

# Mock some URLs if not matching exactly
CORE_TRACE_ASSESSMENTS = [
    ("Occupational Personality Questionnaire OPQ32r", "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/"),
    ("SHL Verify Interactive G+", "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/"),
    ("Graduate Scenarios", "https://www.shl.com/products/product-catalog/view/graduate-scenarios/"),
    ("SVAR Spoken English (US) (New)", "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/"),
    ("Contact Center Call Simulation (New)", "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/"),
    ("Entry Level Customer Serv - Retail & Contact Center", "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-contact-center/"),
    ("Customer Service Phone Simulation", "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/"),
    ("SHL Verify Interactive – Numerical Reasoning", "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/"),
    ("Financial Accounting (New)", "https://www.shl.com/products/product-catalog/view/financial-accounting-new/"),
    ("Basic Statistics (New)", "https://www.shl.com/products/product-catalog/view/basic-statistics-new/"),
    ("Global Skills Assessment", "https://www.shl.com/products/product-catalog/view/global-skills-assessment/"),
    ("Global Skills Development Report", "https://www.shl.com/products/product-catalog/view/global-skills-development-report/"),
    ("OPQ MQ Sales Report", "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/"),
    ("Sales Transformation 2.0 - Individual Contributor", "https://www.shl.com/products/product-catalog/view/sales-transformation-2-0-individual-contributor/"),
    ("Dependability and Safety Instrument (DSI)", "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/"),
    ("Manufac. & Indust. - Safety & Dependability 8.0", "https://www.shl.com/products/product-catalog/view/manufac-indust-safety-dependability-8-0/"),
    ("Workplace Health and Safety (New)", "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/"),
    ("HIPAA (Security)", "https://www.shl.com/products/product-catalog/view/hipaa-security/"),
    ("Medical Terminology (New)", "https://www.shl.com/products/product-catalog/view/medical-terminology-new/"),
    ("Microsoft Word 365 - Essentials (New)", "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/"),
    ("MS Excel (New)", "https://www.shl.com/products/product-catalog/view/ms-excel-new/"),
    ("MS Word (New)", "https://www.shl.com/products/product-catalog/view/ms-word-new/"),
    ("Microsoft Excel 365 (New)", "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/"),
    ("Microsoft Word 365 (New)", "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/"),
    ("Core Java (Advanced Level) (New)", "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/"),
    ("Spring (New)", "https://www.shl.com/products/product-catalog/view/spring-new/"),
    ("RESTful Web Services (New)", "https://www.shl.com/products/product-catalog/view/restful-web-services-new/"),
    ("SQL (New)", "https://www.shl.com/products/product-catalog/view/sql-new/"),
    ("Amazon Web Services (AWS) Development (New)", "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/"),
    ("Docker (New)", "https://www.shl.com/products/product-catalog/view/docker-new/"),
    ("Smart Interview Live Coding", "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/"),
    ("Linux Programming (General)", "https://www.shl.com/products/product-catalog/view/linux-programming-general/"),
    ("Networking and Implementation (New)", "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/"),
    ("OPQ Leadership Report", "https://www.shl.com/products/product-catalog/view/opq-leadership-report/"),
    ("OPQ Universal Competency Report 2.0", "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/")
]

print("Testing resolver:")
for name, url in CORE_TRACE_ASSESSMENTS:
    resolved = resolve_recommendation(name, url)
    if resolved:
        print(f"[OK] '{name}' -> '{resolved[0]}' | Type: {resolved[2]}")
    else:
        print(f"[FAIL] Could NOT resolve: '{name}' (URL: {url})")

"""The app's sample use cases: use cases only, no services or models. The planner picks the services, the Model
Resolver the models. Shown in the sidebar's "Sample use cases" list and pre-built ahead of time (engine/prebuild.py),
so picking a sample opens a finished demo. Each sample's customer name is its project folder (slugified)."""

SAMPLES = {
    "Airline: multilingual live concierge": {
        "customer": "Cymbal Air",
        "ask": "Multilingual real-time voice and avatar concierge for Cymbal Air Rewards members: natural low-latency "
               "conversation in English, Japanese, Spanish, French, German and Chinese, a consistent branded "
               "voice, avatar lip-sync in each of these languages separately, and handoff to human agents."},
    "Retail: real-time fraud detection": {
        "customer": "Acme Retail",
        "ask": "Detect fraudulent card transactions in real time from a stream of 5,000 events per second, score "
               "them with a model, alert analysts within seconds, and keep a queryable history for audits."},
    "Logistics: delivery tracking app": {
        "customer": "Swift Logistics",
        "ask": "Mobile app for customers and drivers with sign-in, live driver location on a map, optimized "
               "delivery routes, push notifications for status changes, and an operations dashboard."},
    "Healthcare: grounded knowledge assistant": {
        "customer": "Global Healthcare Network",
        "ask": "Clinician-facing assistant that answers questions from approved medical guidelines with "
               "citations, extracts structured JSON summaries, and never answers outside the approved corpus."},
    "Hospitality: cinematic video campaign": {
        "customer": "Cymbal Resorts",
        "ask": "Generate a cinematic 1080p promotional video campaign for new resorts with drone-style camera "
               "moves, an original orchestral soundtrack, and brand-safe review before publishing."},
    "Finance: enterprise search with citations": {
        "customer": "Contoso Financial Services",
        "ask": "Enterprise search assistant for employees that answers questions from internal policy documents, "
               "product manuals and wiki pages stored in Google Drive and Cloud Storage, cites the exact source "
               "passage for every answer, respects each employee's document permissions, and says it does not know "
               "when the corpus has no answer."},
    "Commerce: analytics agent with tickets": {
        "customer": "Northwind Commerce",
        "ask": "Analytics agent for the operations team that answers plain-language questions by writing and "
               "running BigQuery SQL over the sales warehouse, explains the result with a chart, detects "
               "week-over-week anomalies, and files a follow-up ticket with the evidence for each anomaly it finds, "
               "asking for confirmation before any write action."},
    "Insurance: claims document extraction": {
        "customer": "Fabrikam Insurance",
        "ask": "Claims intake pipeline that reads scanned claim forms, invoices and photos of damage uploaded by "
               "customers, extracts policy number, dates, amounts and line items into validated structured JSON, "
               "flags missing or inconsistent fields for a human reviewer, and stores the results for audit."},
}

import os

import serpapi
import streamlit as st

# Google Scholar query used for each predicted class
RESEARCH_QUERIES = {
    "Glioma": "glioma brain tumor MRI deep learning classification",
    "Meningioma": "meningioma brain tumor MRI deep learning classification",
    "Pituitary": "pituitary tumor MRI deep learning classification",
    "No Tumor": "brain tumor MRI classification quantum machine learning",
}


def get_serpapi_key():
    key = os.getenv("SERPAPI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["SERPAPI_API_KEY"]
    except Exception:
        return None


# Cached for a day so reruns don't spend SerpApi search credits
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_research_papers(query, _api_key, num_results=5, since_year=2022):
    client = serpapi.Client(api_key=_api_key, timeout=20)
    results = client.search(
        engine="google_scholar",
        q=query,
        as_ylo=since_year,
        num=num_results
    )

    papers = []
    for item in results.get("organic_results", [])[:num_results]:
        pdf_link = next(
            (r.get("link") for r in item.get("resources", [])
             if r.get("file_format") == "PDF" and r.get("link")),
            None
        )
        papers.append({
            "title": item.get("title", "Untitled"),
            "link": item.get("link"),
            "authors": item.get("publication_info", {}).get("summary", ""),
            "snippet": item.get("snippet", ""),
            "cited_by": item.get("inline_links", {}).get("cited_by", {}).get("total"),
            "pdf_link": pdf_link,
        })
    return papers

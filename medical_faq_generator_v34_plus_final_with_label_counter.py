
import streamlit as st
import openai
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from random import choice
from rapidfuzz import fuzz

st.set_page_config(page_title="Medical FAQ Generator (v34+)", layout="wide")
st.title("🧠 Medical FAQ Generator (v34+)")
st.markdown("Uses Google search and manual URLs to generate patient-friendly questions with **token usage tracking**.")

# === STEP 1 ===
if "step" not in st.session_state:
    st.session_state.step = 1

openai_key = st.text_input("🔑 OpenAI API Key", type="password", key="api_key")
google_key = st.text_input("🔑 Google API Key", type="password", key="google_api_key")
google_cx = st.text_input("🔍 Google Search Engine ID (CX)", key="google_cx")
topic = st.text_input("💬 Enter a medical topic", key="topic_input")

if st.session_state.step == 1:
    if all([openai_key, google_key, google_cx, topic]):
        st.session_state.client = openai.OpenAI(api_key=openai_key)
        st.session_state.topic = topic
        st.success("✅ Keys and topic received.")
        st.session_state.step = 2
    else:
        st.info("👆 Please fill in all fields to proceed.")

# === STEP 2 ===
if st.session_state.step == 2:
    st.subheader("🧠 Select Topics and Subtopics")
    predefined_sections = [
        "causes", "symptoms", "diagnosis", "treatment", "patient experiences",
        "FAQs", "real stories", "therapy and recovery", "support groups", "prognosis", "complications"
    ]
    selected_predefined = st.multiselect("✔️ Choose main topics:", predefined_sections)
    custom_topics = st.text_area("➕ Additional topics (one per line):")
    manual_list = [t.strip() for t in custom_topics.splitlines() if t.strip()]
    all_main_topics = selected_predefined + [t for t in manual_list if t.lower() not in map(str.lower, selected_predefined)]

    if all_main_topics:
        subtopics = {}
        for mt in all_main_topics:
            val = st.text_input(f"🔹 Subtopics for '{mt}' (comma-separated)", key=f"sub_{mt}")
            subtopics[mt] = [v.strip() for v in val.split(",") if v.strip()]
    url_limit = st.slider("🔢 Number of URLs to extract per query", key="url_limit_slider", min_value=1, max_value=15, value=3)
    manual_urls = st.text_area("🌐 Manually Add Important URLs (one per line)")
    if st.button("🔎 Run Search"):
            st.session_state.query_settings = {}
            for mt in all_main_topics:
                subs = subtopics.get(mt, [])
                if subs:
                    for s in subs:
                        st.session_state.query_settings[f"{topic} {mt} {s}"] = url_limit
                else:
                    st.session_state.query_settings[f"{topic} {mt}"] = url_limit
            st.session_state.manual_urls = [u.strip() for u in manual_urls.splitlines() if u.strip()]
            st.session_state.step = 3

# === STEP 3 ===
if st.session_state.step == 3:
    st.subheader("🔍 Searching Google + Adding Manual URLs")

    def google_search(query, key, cx, limit):
        urls, start = [], 1
        while len(urls) < limit:
            try:
                url = f"https://www.googleapis.com/customsearch/v1?q={quote(query)}&key={key}&cx={cx}&start={start}"
                res = requests.get(url, timeout=10).json()
                for item in res.get("items", []):
                    link = item.get("link")
                    if link and not any(x in link for x in ["youtube.com", ".pdf", ".gov"]):
                        urls.append(link)
                        if len(urls) == limit:
                            break
                if "items" not in res: break
                start += 10
            except Exception as e:
                st.warning(f"Error in '{query}': {e}")
                break
        return urls

    all_urls = []
    for q, lim in st.session_state.query_settings.items():
        with st.spinner(f"Searching: {q}"):
            current_index = list(st.session_state.query_settings).index(q)
            progress_value = list(st.session_state.query_settings).index(q) / len(st.session_state.query_settings)
            found = google_search(q, google_key, google_cx, lim)
            st.markdown(f"🔗 **{q}** → {len(found)} URLs")
            all_urls.extend(found)
            all_urls.extend(st.session_state.manual_urls)
    before_dedup = len(all_urls)
    all_urls = list(set(all_urls))
    st.info(f"✅ Total URLs after removing {before_dedup - len(all_urls)} duplicates: {len(all_urls)}")
    st.session_state.scraped_urls = all_urls
    st.session_state.step = 4

# === STEP 4 ===
if st.session_state.step == 4:
    st.subheader("🧠 Extracting & Generating Questions")

    def scrape_url(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            return [p.get_text(strip=True) for p in soup.find_all(["p", "li"]) if len(p.get_text(strip=True)) > 30]
        except Exception as e:
            st.warning(f"❌ {url[:60]} – {e}")
            return []

    def chunk_texts(texts, max_len=1200):
        chunks, current = [], ""
        for t in texts:
            if len(current) + len(t) < max_len:
                current += t + " "
            else:
                chunks.append(current.strip())
                current = t + " "
        if current: chunks.append(current.strip())
        return chunks

    def safe_str(val):
        return ", ".join(val) if isinstance(val, list) else str(val)

    def dedup(questions, threshold=80):
        output = []
        for q in questions:
            if all(fuzz.ratio(q, other) < threshold for other in output):
                output.append(q)
        return output

    all_texts = []
    for idx, url in enumerate(st.session_state.scraped_urls):
        st.text(f"🔍 Scraping {idx+1}/{len(st.session_state.scraped_urls)}: {url[:70]}")
        all_texts.extend(scrape_url(url))

    st.info(f"🧾 Extracted {len(all_texts)} text sections from {len(st.session_state.scraped_urls)} URLs")

    if all_texts:
        chunks = chunk_texts(all_texts)
        all_qs, total_tokens = [], 0
        for i, chunk in enumerate(chunks):
            st.text(f"💬 Generating questions from chunk {i+1}/{len(chunks)}")
            prompt = f"""Generate patient-facing questions from this medical text about {st.session_state.topic}:

{chunk}
"""
            try:
                resp = st.session_state.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                msg = resp.choices[0].message.content.strip()
                q_list = [q.strip("-• \n") for q in msg.splitlines() if len(q.strip()) > 10]
                all_qs.extend(q_list)
                total_tokens += resp.usage.total_tokens if resp.usage else 0
            except Exception as e:
                st.warning(f"⚠️ GPT error: {e}")
            time.sleep(1)

        deduped = dedup(all_qs)
        st.success(f"✅ {len(deduped)} unique questions generated after deduplication.")

        labeled = []
        main_topics, subtopics, subsubtopics = [], [], []
        total_tokens = 0
        for i, q in enumerate(deduped):
            try:
                label_prompt = f"""
                st.text(f"🏷️ Labeling {i+1}/{len(deduped)}: {q}")
You are an expert medical knowledge organizer. Given the question below, categorize it into:
- main_topic (e.g., Causes, Symptoms, Diagnosis, Treatment, Prevention, Prognosis, etc.)
- subtopic (e.g., Lifestyle, Medication, Screening, Genetics, etc.)
- optional sub_subtopic (only if appropriate)

Return a JSON object with exactly three fields: "main_topic", "subtopic", and "sub_subtopic".
Use consistent, short, and reusable medical concepts. Do not repeat the full question in the subtopics.
Question: {q}
"""
                resp = st.session_state.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": label_prompt}]
                )
                total_tokens += resp.usage.total_tokens if resp.usage else 0
                raw = eval(resp.choices[0].message.content.strip())
                labeled.append((q, safe_str(raw.get("main_topic")), safe_str(raw.get("subtopic")), safe_str(raw.get("sub_subtopic"))))
            except Exception:
                labeled.append((q, "Uncategorized", "Uncategorized", ""))

        df = pd.DataFrame(labeled, columns=["Generated Questions", "Main Topic", "Subtopic", "Sub-subtopic"])
        filename = f"questions_{st.session_state.topic.replace(' ', '_')}.xlsx"
        df.to_excel(filename, index=False)
        st.success("📥 Questions saved to Excel")
        with open(filename, "rb") as f:
            st.download_button("📥 Download Questions Excel", f, file_name=filename)

        # Show cost estimate
        cost = total_tokens / 1000 * 0.0015  # gpt-3.5-turbo cost per 1K tokens
        st.info(f"💸 Estimated OpenAI token cost: ${cost:.4f} (total tokens: {total_tokens})")

        # Restart option
        if st.button('🔄 Restart App'):
            st.session_state.clear()
            st.experimental_rerun()

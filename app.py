import os

# Must run before huggingface_hub is imported by anything below (directly
# or via langchain_huggingface) since it reads this env var once into a
# module-level constant. Model is already cached locally after first use,
# so this skips slow online freshness checks on every request.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import streamlit as st
import ast
import json
import re

from agents.travel_agent import travel_agent
from rag.vector_store import create_vectorstore


def _normalize_list_line(line):
    return re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip().lower()


def truncate_runaway_repetition(text, max_word_repeats=6, max_line_repeats=3):
    """
    Small local LLMs occasionally degenerate into looping instead of
    stopping: either the same single word forever (e.g. a temple name's
    "Sri" honorific repeated hundreds of times), or the same list item
    repeated verbatim under different numbering (e.g. "33. Temple X"
    ... "49. Temple X"). Everything from the start of such a loop
    onward is unusable, so cut the answer there instead of showing the
    raw repetition to the user.
    """

    cut_at = None

    # Case 1: the same word repeated many times in a row.
    word_pattern = re.compile(
        r"\b(\w+)\b(?:\W+\1\b){" + str(max_word_repeats - 1) + r",}",
        re.IGNORECASE
    )

    word_match = word_pattern.search(text)

    if word_match:
        cut_at = word_match.start()

    # Case 2: the same line/list item repeated verbatim (ignoring
    # leading numbering/bullets), several times in a row.
    lines = text.split("\n")

    line_offsets = []
    pos = 0

    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1

    run_start_idx = None
    run_value = None
    run_len = 0

    for idx, line in enumerate(lines):
        normalized = _normalize_list_line(line)

        if not normalized:
            continue

        if normalized == run_value:
            run_len += 1
        else:
            run_value = normalized
            run_start_idx = idx
            run_len = 1

        if run_len >= max_line_repeats:
            line_cut_at = line_offsets[run_start_idx]

            if cut_at is None or line_cut_at < cut_at:
                cut_at = line_cut_at

            break

    if cut_at is None:
        return text

    truncated = text[:cut_at].rstrip(" \n*_-#")

    if not truncated:
        return (
            "Sorry, I ran into an issue generating that response. "
            "Could you try rephrasing your question?"
        )

    return truncated


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TravelMate AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DARK THEME
# =========================================================

st.markdown(
    """
<style>

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0d12 !important;
    color: #ffffff !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #0b0d12;
}

[data-testid="stHeader"] {
    background-color: #0b0d12;
}

[data-testid="stToolbar"] {
    background-color: #0b0d12;
}


/* =========================
   MAIN
   ========================= */

.main .block-container {
    max-width: 1250px;
    padding-top: 35px;
    padding-bottom: 100px;
}


/* =========================
   SIDEBAR
   ========================= */

[data-testid="stSidebar"] {
    background-color: #10131a !important;
    border-right: 1px solid #242832;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.sidebar-logo {
    font-size: 25px;
    font-weight: 700;
    color: #ffffff;
}

.sidebar-subtitle {
    color: #8b93a7;
    font-size: 14px;
    margin-bottom: 25px;
}

.sidebar-heading {
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}


/* =========================
   HERO
   ========================= */

.hero {
    background: linear-gradient(
        135deg,
        #151923,
        #10131a
    );

    border: 1px solid #292f3b;
    border-radius: 24px;

    padding: 38px 42px;

    margin-bottom: 24px;

    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 750;
    color: #ffffff;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 16px;
    color: #9ca3af;
    line-height: 1.6;
}


/* =========================
   STATUS
   ========================= */

.agent-status {
    background: #111b17;
    border: 1px solid #1e4d39;

    border-radius: 15px;

    padding: 14px 18px;

    color: #86efac;

    margin-bottom: 30px;

    font-size: 14px;
}


/* =========================
   SECTION TITLE
   ========================= */

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #ffffff;

    margin-top: 25px;
    margin-bottom: 18px;
}


/* =========================
   FEATURE CARDS
   ========================= */

.feature-card {
    background: #11141b;

    border: 1px solid #272d38;

    border-radius: 18px;

    padding: 23px;

    min-height: 145px;

    box-shadow: 0 5px 20px rgba(0,0,0,0.15);
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 17px;
    font-weight: 700;
    color: #ffffff;
}

.feature-text {
    font-size: 13px;
    color: #8b93a7;
    margin-top: 7px;
    line-height: 1.5;
}


/* =========================
   WEATHER CARD
   ========================= */

.weather-card {
    background: linear-gradient(
        135deg,
        #151923,
        #11141b
    );

    border: 1px solid #303744;

    border-radius: 20px;

    padding: 25px;

    margin: 15px 0;

    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

.weather-city {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}

.weather-temp {
    font-size: 42px;
    font-weight: 750;
    color: #ffffff;
    margin-top: 12px;
}

.weather-label {
    color: #8b93a7;
    font-size: 13px;
}

.weather-value {
    color: #e5e7eb;
    font-size: 16px;
    font-weight: 600;
}


/* =========================
   PLACE CARDS
   ========================= */

.place-card {
    background: #11141b;

    border: 1px solid #272d38;

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 12px;

    transition: 0.2s;
}

.place-card:hover {
    border-color: #4b5563;
}

.place-name {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

.place-info {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.7;
}


/* =========================
   RAG CARD
   ========================= */

.rag-card {
    background: #11141b;

    border: 1px solid #303744;

    border-radius: 18px;

    padding: 22px;

    margin: 15px 0;

    color: #d1d5db;

    line-height: 1.7;
}


/* =========================
   CHAT
   ========================= */

[data-testid="stChatMessage"] {
    background-color: #11141b;
    border: 1px solid #242a35;
    border-radius: 18px;
    padding: 10px;
    margin-bottom: 10px;
}


/* =========================
   INPUT
   ========================= */

[data-testid="stChatInput"] textarea {
    background-color: #171a22 !important;
    color: #ffffff !important;

    border: 1px solid #303744 !important;

    border-radius: 15px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #737b8c !important;
}


/* =========================
   BUTTON
   ========================= */

.stButton button {
    background-color: #171a22;
    color: #ffffff;

    border: 1px solid #303744;

    border-radius: 12px;

    font-weight: 600;
}

.stButton button:hover {
    background-color: #222733;
    border-color: #4b5563;
}


/* =========================
   ALERT
   ========================= */

[data-testid="stAlert"] {
    background-color: #11141b;
    border: 1px solid #292f3b;
    color: #d1d5db;
}


/* =========================
   DIVIDER
   ========================= */

hr {
    border-color: #252a34 !important;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">✈️ TravelMate AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Your intelligent travel companion'
        '</div>',
        unsafe_allow_html=True
    )

    # =========================
# Travel Documents
# =========================

    st.sidebar.subheader("📄 Travel Documents")

    st.sidebar.write(
        "Upload travel documents to enhance the AI's knowledge."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "xlsx", "png", "jpg", "jpeg"],
        help="Supported formats: PDF, DOCX, TXT, XLSX, PNG, JPG, JPEG",
        key="travel_document_uploader"
    )

    st.sidebar.caption("200 MB per file • PDF, DOCX, TXT, XLSX, Images")

    if uploaded_file is not None:

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process only when a new document is uploaded
        if (
            "uploaded_file_name" not in st.session_state
            or st.session_state.uploaded_file_name != uploaded_file.name
        ):

            with st.spinner("Processing your travel document..."):

                try:
                    _, processed_docs = create_vectorstore(file_path)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success("Travel document processed successfully! 📚")

                    if processed_docs and processed_docs[0].metadata.get("type") == "image":
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"🖼️ {processed_docs[0].page_content}"
                            }
                        )

                except Exception as e:
                    st.error(f"Couldn't process this file: {e}")

        else:
            st.info("This document is already added to the knowledge base.")

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">🧭 EXPLORE</div>',
        unsafe_allow_html=True
    )
    st.write("🌦️ Weather")
    st.write("📍 Places & Restaurants")
    st.write("🏨 Hotels")
    st.write("🏛️ Tourist Attractions")
    st.write("📄 Travel Documents")

    st.markdown(
        '<div class="sidebar-heading">🤖 AI CAPABILITIES</div>',
        unsafe_allow_html=True
    )

    st.write("🔎 Semantic Search")
    st.write("🧠 RAG Question Answering")
    st.write("🤖 AI Travel Agent")

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">⚙️ CHAT</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption("RAG + LangGraph + AI Agent")


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">✈️ TravelMate AI</div>
<div class="hero-subtitle">
Your AI-powered travel companion for discovering places,
checking weather, finding restaurants, and answering travel questions.
</div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# STATUS
# =========================================================

st.markdown(
    """
<div class="agent-status">
🟢 <b>AI Travel Agent Ready</b>
&nbsp;&nbsp;•&nbsp;&nbsp;
RAG Search Available
&nbsp;&nbsp;•&nbsp;&nbsp;
Weather Available
&nbsp;&nbsp;•&nbsp;&nbsp;
Places Search Available
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# FEATURE CARDS
# =========================================================

st.markdown(
    '<div class="section-title">What can I help you with?</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🌦️</div>
<div class="feature-title">Weather</div>
<div class="feature-text">
Check current weather conditions for your destination.
</div>
</div>
""",
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🍴</div>
<div class="feature-title">Restaurants</div>
<div class="feature-text">
Find restaurants and food places around your destination.
</div>
</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">📍</div>
<div class="feature-title">Places</div>
<div class="feature-text">
Discover tourist attractions and interesting places.
</div>
</div>
""",
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">📄</div>
<div class="feature-title">Travel Knowledge</div>
<div class="feature-text">
Ask questions from your uploaded travel documents.
</div>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# CHAT
# =========================================================

st.markdown(
    '<div class="section-title">💬 Travel Assistant</div>',
    unsafe_allow_html=True
)


if len(st.session_state.messages) == 0:

    st.info(
        "👋 Welcome to TravelMate AI! "
        "Try asking: **What is the weather in Hyderabad?**"
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# RENDER WEATHER
# =========================================================

def display_weather(data):

    st.markdown(
        f"""
<div class="weather-card">

<div class="weather-city">
🌦️ {data.get("city", "Unknown")}, {data.get("country", "")}
</div>

<div class="weather-temp">
{data.get("temperature", "--")}°C
</div>

<div class="weather-label">
Current Temperature
</div>

<br>

<div>
💧 <b>Humidity:</b>
{data.get("humidity", "--")}%
</div>

<div>
🌡️ <b>Feels Like:</b>
{data.get("apparent_temperature", "--")}°C
</div>

<div>
💨 <b>Wind Speed:</b>
{data.get("wind_speed", "--")} km/h
</div>

</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# RENDER PLACES
# =========================================================

def display_places(places):
    st.markdown(
        '<div class="section-title">🍴 Recommended Places</div>',
        unsafe_allow_html=True
    )

    # Remove duplicate places
    unique_places = []
    seen = set()

    for place in places:
        name = place.get("name", "").strip().lower()

        if name and name not in seen:
            seen.add(name)
            unique_places.append(place)

    columns = st.columns(2)

    for index, place in enumerate(unique_places):
        column = columns[index % 2]

        with column:
            st.markdown(
                f"""
                <div class="place-card">
                <div class="place-name">
                📍 {place.get("name", "Unknown Place")}
                </div>

                <div class="place-info">
                📌 {place.get("address", "Address unavailable")}
                </div>

                <div class="place-info">
                📞 {place.get("phone", "Not available")}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PARSE TOOL RESULT
# =========================================================

def parse_tool_content(content):

    if isinstance(content, (dict, list)):

        return content

    if not isinstance(content, str):

        return content

    try:

        return json.loads(content)

    except Exception:
        pass

    try:

        return ast.literal_eval(content)

    except Exception:
        return content


# =========================================================
# DISPLAY TOOL RESULTS
# =========================================================

def display_tool_result(tool_name, content):

    data = parse_tool_content(content)

    # -----------------------------------------
    # WEATHER
    # -----------------------------------------

    if tool_name == "weather_tool":

        if isinstance(data, dict):

            display_weather(data)

        else:

            st.markdown(str(data))

        return


    # -----------------------------------------
    # PLACES
    # -----------------------------------------

    if tool_name == "places_tool":

        if isinstance(data, list):

            display_places(data)

        else:

            st.markdown(str(data))

        return


    # -----------------------------------------
    # RAG
    # -----------------------------------------

    if tool_name == "travel_knowledge_tool":

        st.markdown(
            """
<div class="rag-card">
<b>📚 Travel Knowledge</b>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(str(data))

        return


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask about weather, restaurants, places, or travel documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------
    # USER MESSAGE
    # -----------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -----------------------------------------
    # AI AGENT
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "✈️ TravelMate is planning your answer..."
        ):

            response = travel_agent.invoke(
                {
                    "messages": st.session_state.messages
                }
            )


               # -------------------------------------
        # Process every message returned
        # -------------------------------------

        final_answer = None

        for msg in response["messages"]:

            # -------------------------------------
            # Tool message
            # -------------------------------------

            if getattr(msg, "type", None) == "tool":

                tool_name = getattr(
                    msg,
                    "name",
                    ""
                )

                # Show UI only for weather and places
                if tool_name in [
                    "weather_tool",
                    "places_tool"
                ]:

                    display_tool_result(
                        tool_name,
                        msg.content
                    )

                # Hide raw RAG tool output
                elif tool_name == "travel_knowledge_tool":

                    pass

            # -------------------------------------
            # AI message
            # -------------------------------------

            elif getattr(msg, "type", None) == "ai":

                content = msg.content

                if isinstance(content, str):

                    if content.strip():

                        final_answer = content

                elif isinstance(content, list):

                    text_parts = []

                    for block in content:

                        if isinstance(block, dict):

                            if block.get("type") == "text":

                                text_parts.append(
                                    block.get("text", "")
                                )

                        elif isinstance(block, str):

                            text_parts.append(block)

                    final_answer = "\n".join(
                        text_parts
                    ).strip()

        # -------------------------------------
        # Display final response
        # -------------------------------------

        if final_answer:
            final_answer = final_answer.replace("<br>", "\n")
            final_answer = final_answer.replace("<br/>", "\n")
            final_answer = final_answer.replace("<br />", "\n")
            final_answer = truncate_runaway_repetition(final_answer)

        st.markdown(final_answer)

        # -------------------------------------
        # Save final response
        # -------------------------------------

        if final_answer:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer
                }
            )

        else:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I found the information you requested."
                }
            )
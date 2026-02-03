import re
import sqlite3
from dataclasses import dataclass
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import List, Dict, Tuple

import streamlit as st

# Optional tooling
SKLEARN_AVAILABLE = False
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

PANDAS_AVAILABLE = False
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Feedback Analyzer", layout="wide")

DB_PATH = "feedback.db"


# -----------------------------
# Pattern-based lexicons (subtypes)
# -----------------------------
def compile_patterns(patterns):
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def count_regex_hits(text: str, compiled_patterns) -> int:
    if not text:
        return 0
    hits = 0
    for pat in compiled_patterns:
        hits += len(pat.findall(text))
    return hits


PRAISE_PATTERNS = {
    "praise_general": compile_patterns([
        r"\b(excellent|outstanding|great|fantastic|superb|remarkable|impressive)\b",
        r"\b(well done|nice work|good work|excellent job|strong work)\b",
        r"\b(keep it up|well executed|very good)\b",
    ]),
    "praise_clarity": compile_patterns([
        r"\b(clear|clearly|coherent|well[- ]structured|well[- ]organized|easy to follow|well written|concise)\b",
        r"\b(well explained|clearly articulated|good structure|strong flow)\b",
    ]),
    "praise_depth": compile_patterns([
        r"\b(insightful|thoughtful|nuanced|sophisticated|mature|deep|in-depth)\b",
        r"\b(good insight|strong analysis|careful analysis|excellent analysis)\b",
    ]),
    "praise_evidence": compile_patterns([
        r"\b(well supported|well grounded|good use of evidence|strong evidence)\b",
        r"\b(appropriate references|effective use of literature|good citations|well referenced)\b",
    ]),
    "praise_effort": compile_patterns([
        r"\b(thorough|comprehensive|detailed|carefully done|well researched)\b",
        r"\b(shows effort|good effort|strong effort)\b",
    ]),
}

CRITIQUE_PATTERNS = {
    "critique_general": compile_patterns([
        r"\b(needs|need to|should|could|would benefit|requires|lacks|missing|incomplete|underdeveloped)\b",
        r"\b(improve|strengthen|revise|refine|rework|tighten|reconsider)\b",
        r"\b(however|but|although|yet|nevertheless)\b",
    ]),
    "critique_clarity": compile_patterns([
        r"\b(unclear|confusing|ambiguous|vague|hard to follow|not clear)\b",
        r"\b(clarify|clarification|clarifying|explain|explanation|expand on)\b",
        r"\b(define|definition|be specific|more specific)\b",
    ]),
    "critique_argument": compile_patterns([
        r"\b(unsupported|unconvincing|weak argument|not justified|questionable)\b",
        r"\b(justify|justification|defend|support this claim|evidence needed)\b",
        r"\b(assumption|assumes|asserts)\b",
    ]),
    "critique_depth": compile_patterns([
        r"\b(superficial|oversimplified|too general|glosses over)\b",
        r"\b(needs more depth|more depth|develop further|go deeper|drill into)\b",
        r"\b(more analysis|analyze further)\b",
    ]),
    "critique_evidence": compile_patterns([
        r"\b(cite|citation|reference|source needed|needs sources)\b",
        r"\b(evidence|support with evidence|back this up)\b",
        r"\b(engage with the literature|review the literature|read more)\b",
    ]),
    "critique_structure": compile_patterns([
        r"\b(structure|organization|flow|transition|signpost)\b",
        r"\b(repetition|redundant|reorganize|unclear structure)\b",
    ]),
    "critique_questions": compile_patterns([r"\?"]),
}


@dataclass
class Row:
    section: str
    comment: str
    score: float


# -----------------------------
# DB layer (SQLite)
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT NOT NULL,
            comment TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def db_count() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    n = cur.fetchone()[0]
    conn.close()
    return int(n)


def db_insert(row: Row) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback (section, comment, score) VALUES (?, ?, ?)",
        (row.section.strip(), row.comment.strip(), float(row.score)),
    )
    conn.commit()
    conn.close()


def db_fetch_all() -> List[Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT section, comment, score FROM feedback ORDER BY id ASC")
    rows = [Row(section=r[0], comment=r[1], score=float(r[2])) for r in cur.fetchall()]
    conn.close()
    return rows


def db_delete_all() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM feedback")
    conn.commit()
    conn.close()


# -----------------------------
# Text feature helpers
# -----------------------------
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", (text or "").lower()))


def sentence_count(text: str) -> int:
    s = re.split(r"[.!?]+\s*", (text or "").strip())
    s = [x for x in s if x]
    return len(s)


def question_count(text: str) -> int:
    return (text or "").count("?")


def subtype_scores(text: str) -> dict:
    out = {}
    for name, pats in PRAISE_PATTERNS.items():
        out[name] = count_regex_hits(text, pats)
    for name, pats in CRITIQUE_PATTERNS.items():
        out[name] = count_regex_hits(text, pats)
    return out


def aggregate_scores(score_dict: dict) -> dict:
    praise_total = sum(v for k, v in score_dict.items() if k.startswith("praise_"))
    critique_total = sum(v for k, v in score_dict.items() if k.startswith("critique_"))
    ratio_pc = (praise_total / critique_total) if critique_total > 0 else float("inf")
    score_dict["praise_total"] = praise_total
    score_dict["critique_total"] = critique_total
    score_dict["praise_to_critique_ratio"] = ratio_pc
    return score_dict


def grade_letter(score: float) -> str:
    if score >= 97: return "A+"
    if score >= 92: return "A"
    if score >= 87: return "A-"
    if score >= 80: return "B+"
    if score >= 75: return "B"
    if score >= 71: return "B-"
    if score >= 67: return "C+"
    if score >= 64: return "C"
    if score >= 60: return "C-"
    return "F"


def summarize_scores(scores: List[float]) -> dict:
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": mean(scores),
        "median": median(scores),
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def section_report(rows: List[Row]) -> dict:
    scores = [r.score for r in rows]
    comments = [r.comment for r in rows]
    grades = [grade_letter(s) for s in scores]

    wc = [word_count(c) for c in comments]
    sc = [sentence_count(c) for c in comments]
    qc = [question_count(c) for c in comments]

    per_comment = [aggregate_scores(subtype_scores(c)) for c in comments]

    base = summarize_scores(scores)
    base.update(
        {
            "grade_dist": dict(Counter(grades)),
            "avg_words": mean(wc) if wc else 0,
            "avg_sentences": mean(sc) if sc else 0,
            "avg_questions": mean(qc) if qc else 0,
        }
    )

    if per_comment:
        keys = per_comment[0].keys()
        for k in keys:
            vals = [d.get(k, 0) for d in per_comment]
            base[f"avg_{k}"] = mean(vals)

    return base


def tfidf_top_terms(rows: List[Row], top_k: int = 12) -> List[Tuple[str, float]]:
    if not SKLEARN_AVAILABLE or len(rows) < 5:
        return []
    texts = [r.comment for r in rows]
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(texts)
    avg = np.asarray(X.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    pairs = sorted(zip(terms, avg), key=lambda x: x[1], reverse=True)[:top_k]
    return [(t, float(s)) for t, s in pairs]


def safe_ml_sanity_check(rows: List[Row]) -> dict:
    if not SKLEARN_AVAILABLE or len(rows) < 30:
        return {"enabled": False, "reason": "sklearn missing or too few samples"}

    texts = [r.comment for r in rows]
    y = np.array([r.score for r in rows], dtype=float)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)

    model = Ridge(alpha=1.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2 = cross_val_score(model, X, y, cv=kf, scoring="r2").mean()
    mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()

    return {"enabled": True, "cv_r2": float(r2), "cv_mae": float(mae)}


# -----------------------------
# Explanation helpers (NEW)
# -----------------------------
def fmt(x: float, nd=2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def safe_get(d: dict, k: str, default=0.0):
    return d.get(k, default)


def interpret_ratio(ratio: float) -> str:
    if ratio == float("inf"):
        return "Feedback is almost entirely positive (little or no critique)."
    if ratio >= 2.0:
        return "Feedback is praise-heavy (more praise than critique)."
    if ratio >= 1.0:
        return "Feedback is balanced (similar amounts of praise and critique)."
    if ratio >= 0.5:
        return "Feedback is critique-leaning (more critique than praise)."
    return "Feedback is strongly critique-heavy."


def interpret_questions(avg_q: float) -> str:
    if avg_q >= 2.0:
        return "Feedback contains many questions, which often signals confusion or missing detail."
    if avg_q >= 1.0:
        return "Feedback regularly asks questions; consider adding clarity or examples."
    if avg_q > 0:
        return "Some questions appear; clarity may be uneven."
    return "Almost no questions; feedback is mostly statements."


def interpret_variability(std: float) -> str:
    if std >= 12:
        return "Scores vary a lot (inconsistent quality across entries)."
    if std >= 7:
        return "Scores vary moderately (some inconsistency)."
    return "Scores are fairly consistent."


def pick_top_bottom_sections(by_section: dict) -> Tuple[List[Tuple[str, float, int]], List[Tuple[str, float, int]]]:
    sec_means = []
    for sec, rws in by_section.items():
        rep = section_report(rws)
        if rep.get("n", 0) > 0:
            sec_means.append((sec, rep["mean"], rep["n"]))
    sec_means.sort(key=lambda x: x[1], reverse=True)
    top = sec_means[:3]
    bottom = sec_means[-3:] if len(sec_means) > 3 else []
    return top, bottom


def narrative_summary(title: str, rep: dict) -> str:
    n = rep.get("n", 0)
    if n == 0:
        return "No data yet."

    ratio = safe_get(rep, "avg_praise_to_critique_ratio", 0.0)
    std = safe_get(rep, "std", 0.0)
    avg_q = safe_get(rep, "avg_questions", 0.0)

    critique_keys = [
        "avg_critique_clarity",
        "avg_critique_argument",
        "avg_critique_depth",
        "avg_critique_evidence",
        "avg_critique_structure",
        "avg_critique_questions",
    ]
    praise_keys = [
        "avg_praise_clarity",
        "avg_praise_depth",
        "avg_praise_evidence",
        "avg_praise_effort",
        "avg_praise_general",
    ]

    top_crit = max(critique_keys, key=lambda k: safe_get(rep, k, 0.0))
    top_praise = max(praise_keys, key=lambda k: safe_get(rep, k, 0.0))

    map_name = {
        "avg_critique_clarity": "clarity issues",
        "avg_critique_argument": "argument/justification issues",
        "avg_critique_depth": "lack of depth",
        "avg_critique_evidence": "missing evidence/citations",
        "avg_critique_structure": "structure/flow issues",
        "avg_critique_questions": "open questions / uncertainty",
        "avg_praise_clarity": "clarity and readability",
        "avg_praise_depth": "depth and insight",
        "avg_praise_evidence": "evidence and references",
        "avg_praise_effort": "effort and thoroughness",
        "avg_praise_general": "general strong quality",
    }

    return (
        f"**{title}** has **{n}** entries with an average score of **{fmt(rep['mean'])}** "
        f"(median **{fmt(rep['median'])}**, std **{fmt(std)}**). "
        f"{interpret_variability(std)}\n\n"
        f"Feedback tone: **{interpret_ratio(ratio)}** "
        f"(praise/critique ratio ≈ **{fmt(ratio)}**). "
        f"{interpret_questions(avg_q)}\n\n"
        f"Most common strength signal: **{map_name.get(top_praise, top_praise)}**. "
        f"Most common improvement signal: **{map_name.get(top_crit, top_crit)}**."
    )


def bar_chart_explanation(df, metric: str) -> str:
    if df is None or metric not in df.columns:
        return ""

    nice = {
        "score": "Score (0–100)",
        "praise_total": "Praise frequency",
        "critique_total": "Critique frequency",
        "critique_questions": "Question marks (uncertainty/clarification prompts)",
    }.get(metric, metric)

    return (
        f"This bar chart shows the **average {nice} per section** (with error bars showing variation). "
        f"Sections with higher bars have higher average values for this metric. "
        f"If a section has large error bars, it means the entries in that section are inconsistent."
    )


def heatmap_explanation(df, metrics: list, agg: str) -> str:
    if df is None:
        return ""

    table = df.groupby("section")[metrics].mean() if agg == "mean" else df.groupby("section")[metrics].median()

    highlights = []
    for m in ["score", "praise_total", "critique_total", "critique_questions"]:
        if m in table.columns:
            best = table[m].idxmax()
            worst = table[m].idxmin()
            highlights.append((m, best, float(table.loc[best, m]), worst, float(table.loc[worst, m])))

    name = {
        "score": "Score",
        "praise_total": "Praise",
        "critique_total": "Critique",
        "critique_questions": "Questions",
    }

    lines = [
        f"The heatmap summarizes **sections × metrics** using the **{agg}**. Darker/brighter cells indicate larger values.",
        "Key highlights:",
    ]
    for m, bsec, bval, wsec, wval in highlights:
        lines.append(f"- **{name.get(m, m)}**: highest in **Section {bsec}** ({fmt(bval)}), lowest in **Section {wsec}** ({fmt(wval)})")
    return "\n".join(lines)


def recommendations(overall_rep: dict, by_section: dict) -> List[str]:
    recs = []
    ratio = safe_get(overall_rep, "avg_praise_to_critique_ratio", 0.0)

    if ratio == float("inf") or ratio >= 2.5:
        recs.append("Add a rubric/checklist so feedback includes at least 1 concrete improvement point per entry.")

    if safe_get(overall_rep, "avg_questions", 0.0) >= 1.0:
        recs.append("High question rate usually means unclear writing — add definitions, examples, or step-by-step explanation.")

    top, bottom = pick_top_bottom_sections(by_section)
    if bottom:
        low_secs = ", ".join([str(s) for s, _, _ in bottom])
        recs.append(f"Prioritize improving these lower-performing sections first: **{low_secs}**.")

    if safe_get(overall_rep, "avg_critique_evidence", 0.0) >= 0.8:
        recs.append("Evidence/citations are a recurring issue — add 1–2 references per claim or include a short sources paragraph.")

    if not recs:
        recs.append("Overall looks consistent. Next step: track results over time (weekly) to see whether revisions improve specific sections.")

    return recs


# -----------------------------
# UI helpers
# -----------------------------
def section_sort_key(x: str):
    s = str(x).strip()
    return (0, int(s)) if s.isdigit() else (1, s.lower())


def render_report(title: str, rows: List[Row]):
    rep = section_report(rows)

    st.subheader(title)
    if rep.get("n", 0) == 0:
        st.info("No data yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", rep["n"])
    c2.metric("Mean", f"{rep['mean']:.2f}")
    c3.metric("Median", f"{rep['median']:.2f}")
    c4.metric("Std", f"{rep['std']:.2f}")

    st.write(
        f"**Min/Max:** {rep['min']:.2f} / {rep['max']:.2f}  |  "
        f"**Avg words:** {rep['avg_words']:.1f}  |  "
        f"**Avg sentences:** {rep['avg_sentences']:.1f}  |  "
        f"**Avg questions:** {rep['avg_questions']:.2f}"
    )

    st.write("**Grade distribution:**", rep["grade_dist"])

    with st.expander("Praise/Critique breakdown"):
        st.write(
            f"- Avg praise_total: **{rep.get('avg_praise_total', 0):.2f}**\n"
            f"- Avg critique_total: **{rep.get('avg_critique_total', 0):.2f}**\n"
            f"- Avg praise/critique ratio: **{rep.get('avg_praise_to_critique_ratio', 0):.2f}**"
        )
        st.write(
            f"**Praise** — clarity: {rep.get('avg_praise_clarity', 0):.2f}, "
            f"depth: {rep.get('avg_praise_depth', 0):.2f}, "
            f"evidence: {rep.get('avg_praise_evidence', 0):.2f}, "
            f"effort: {rep.get('avg_praise_effort', 0):.2f}"
        )
        st.write(
            f"**Critique** — clarity: {rep.get('avg_critique_clarity', 0):.2f}, "
            f"argument: {rep.get('avg_critique_argument', 0):.2f}, "
            f"depth: {rep.get('avg_critique_depth', 0):.2f}, "
            f"evidence: {rep.get('avg_critique_evidence', 0):.2f}, "
            f"structure: {rep.get('avg_critique_structure', 0):.2f}, "
            f"questions: {rep.get('avg_critique_questions', 0):.2f}"
        )

    if SKLEARN_AVAILABLE:
        terms = tfidf_top_terms(rows, top_k=10)
        if terms:
            with st.expander("Top TF-IDF terms (what feedback focuses on)"):
                st.write(", ".join([f"**{t}**" for t, _ in terms]))

        ml = safe_ml_sanity_check(rows)
        with st.expander("ML sanity-check (TF-IDF + Ridge)"):
            if ml.get("enabled"):
                st.write(f"- CV R²: **{ml['cv_r2']:.3f}**")
                st.write(f"- CV MAE: **{ml['cv_mae']:.2f}** points")
            else:
                st.write(f"Disabled: {ml.get('reason')}")


def make_dataframe(rows: List[Row]):
    if not PANDAS_AVAILABLE:
        return None
    records = []
    for r in rows:
        feats = {
            "section": r.section,
            "score": r.score,
            "grade": grade_letter(r.score),
            "word_count": word_count(r.comment),
            "sentence_count": sentence_count(r.comment),
            "question_count": question_count(r.comment),
        }
        subs = aggregate_scores(subtype_scores(r.comment))
        feats.update(subs)
        records.append(feats)
    return pd.DataFrame(records)


def plot_section_bars(df, metric="score"):
    grp = df.groupby("section")[metric]
    means = grp.mean()
    stds = grp.std().fillna(0)

    fig, ax = plt.subplots()
    ax.bar(means.index.astype(str), means.values, yerr=stds.values, capsize=4)
    ax.set_xlabel("Section")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Per-section mean {metric} (±1 std)")
    fig.tight_layout()
    return fig


def plot_heatmap(df, metrics, agg="mean"):
    if agg == "mean":
        table = df.groupby("section")[metrics].mean()
    else:
        table = df.groupby("section")[metrics].median()

    data = table.values

    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index.astype(str))
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_title(f"Section x Metric heatmap ({agg})")

    if data.shape[0] <= 12 and data.shape[1] <= 12:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    return fig


# -----------------------------
# App
# -----------------------------
init_db()

st.title("📘 Feedback Dataset Analyzer (Web Version)")
st.caption("Add feedback, analyze overall + per section, and visualize patterns. Uses SQLite so deleting files won’t crash the app.")

# Sidebar
with st.sidebar:
    st.header("Dataset")
    st.write(f"**Records:** {db_count()}")
    st.markdown("---")

    st.subheader("Reset (Delete all data)")
    confirm = st.checkbox("I understand this will delete everything.")
    if st.button("🧨 Delete all records", type="primary", disabled=not confirm):
        db_delete_all()
        st.success("All data deleted.")
        st.rerun()

rows = db_fetch_all()

tab_add, tab_analyze, tab_visualize, tab_export = st.tabs(["➕ Add Entry", "📊 Analyze", "📈 Visualize", "⬇️ Export"])

with tab_add:
    st.subheader("Add a new entry")

    with st.form("add_form", clear_on_submit=True):
        section = st.text_input("Section (e.g., 1, 2, 3, or 'Intro')", value="1")
        comment = st.text_area("Comment", height=180, placeholder="Paste feedback here...")
        score = st.number_input("Score (0–100)", min_value=0.0, max_value=100.0, value=85.0, step=0.5)
        submitted = st.form_submit_button("Save")

    if submitted:
        if not comment.strip():
            st.error("Comment cannot be empty.")
        else:
            db_insert(Row(section=section, comment=comment, score=float(score)))
            st.success("Saved!")
            st.rerun()

with tab_analyze:
    st.subheader("Summary")

    if not rows:
        st.info("No data yet. Add entries in the first tab.")
    else:
        # Overall
        render_report("Overall", rows)

        with st.expander("📌 Glossary (what these metrics mean)"):
            st.write("""
- **Score**: Your numeric rating (0–100).
- **Praise total**: Counts of positive phrases like “excellent”, “well done”, etc.
- **Critique total**: Counts of improvement language like “needs”, “should”, “unclear”, etc.
- **Praise/Critique ratio**: Higher means more praise-heavy; lower means more critique-heavy.
- **Questions**: Number of “?” — often indicates uncertainty, confusion, or requests for clarification.
- **Std (standard deviation)**: Higher means scores are inconsistent; lower means consistent.
""")

        st.markdown("## 🧠 Plain-English Explanation")
        overall_rep = section_report(rows)

        by_section: Dict[str, List[Row]] = defaultdict(list)
        for r in rows:
            by_section[r.section].append(r)

        st.info(f"Tone summary: {interpret_ratio(safe_get(overall_rep, 'avg_praise_to_critique_ratio', 0.0))}")
        st.write(narrative_summary("Overall", overall_rep))

        top, bottom = pick_top_bottom_sections(by_section)
        if top:
            st.markdown("### ⭐ Strongest sections (by mean score)")
            for sec, m, n in top:
                st.write(f"- **Section {sec}** — mean **{fmt(m)}** (N={n})")
        if bottom:
            st.markdown("### ⚠️ Weakest sections (by mean score)")
            for sec, m, n in bottom:
                st.write(f"- **Section {sec}** — mean **{fmt(m)}** (N={n})")

        st.markdown("### ✅ Actionable Recommendations")
        for r in recommendations(overall_rep, by_section):
            st.write(f"- {r}")

        # Per section
        st.markdown("---")
        st.subheader("Per-section")
        for sec in sorted(by_section.keys(), key=section_sort_key):
            render_report(f"Section {sec}", by_section[sec])
            rep = section_report(by_section[sec])
            st.write(narrative_summary(f"Section {sec}", rep))

with tab_visualize:
    st.subheader("Charts")

    if not PANDAS_AVAILABLE:
        st.warning("pandas/matplotlib not available. Add them to requirements.txt to enable plots.")
    elif not rows:
        st.info("No data yet.")
    else:
        df = make_dataframe(rows)
        st.dataframe(df[["section", "score", "grade", "word_count", "sentence_count", "question_count"]], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox("Metric (bar chart)", ["score", "praise_total", "critique_total", "critique_questions"])
        with col2:
            agg = st.selectbox("Heatmap aggregation", ["mean", "median"])

        fig1 = plot_section_bars(df, metric=metric)
        st.pyplot(fig1, clear_figure=True)

        metrics = [
            "score",
            "praise_total",
            "critique_total",
            "praise_to_critique_ratio",
            "praise_clarity",
            "praise_depth",
            "critique_clarity",
            "critique_depth",
            "critique_evidence",
            "critique_questions",
        ]
        fig2 = plot_heatmap(df, metrics=metrics, agg=agg)
        st.pyplot(fig2, clear_figure=True)

        st.markdown("## 🧾 How to read these figures")
        st.write(bar_chart_explanation(df, metric))
        st.write(heatmap_explanation(df, metrics=metrics, agg=agg))

with tab_export:
    st.subheader("Export your data")

    if not PANDAS_AVAILABLE:
        st.warning("pandas not available. Add it to requirements.txt to export nicely.")
    elif not rows:
        st.info("No data to export.")
    else:
        df_export = pd.DataFrame([{"Section": r.section, "Comment": r.comment, "Score": r.score} for r in rows])
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="feedback_dataset_export.csv", mime="text/csv")
        st.write("Tip: This export is useful if you ever migrate to Google Sheets / Supabase / Postgres later.")
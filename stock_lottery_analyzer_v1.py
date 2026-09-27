import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Stock Lottery Analyzer v1", page_icon="🎯", layout="wide")

# ============================================================
# STOCK LOTTERY ANALYZER v1
# 6 markets: A/B/C x AM/PM
# Numeric results are stored as strings to preserve leading zeros.
# ============================================================

COLUMNS = [
    "date", "market", "session", "three_digit",
    "two_digit_top", "two_digit_bottom", "source", "verified"
]

MARKETS = {
    "A": "นิเคอิ",
    "B": "หุ้นไทย",
    "C": "ฮั่งเส็ง",
}
SESSIONS = {"AM": "เช้า", "PM": "บ่าย"}

# Initial dataset: only rows supplied/confirmed in the working conversation.
# More rows can be appended later without changing the analyzer.
INITIAL_ROWS = [
    ["2026-09-25","B","AM","358","58","00","working dataset","VERIFIED"],
    ["2026-09-24","B","AM","534","34","36","working dataset","VERIFIED"],
    ["2026-09-23","B","AM","404","04","40","working dataset","VERIFIED"],
    ["2026-09-22","B","AM","345","45","42","working dataset","VERIFIED"],
    ["2026-09-21","B","AM","322","22","07","working dataset","VERIFIED"],
    ["2026-09-18","B","AM","036","36","02","working dataset","VERIFIED"],
    ["2026-09-17","B","AM","166","66","93","working dataset","VERIFIED"],
    ["2026-09-16","B","AM","169","69","03","working dataset","VERIFIED"],
    ["2026-09-15","B","AM","050","50","57","working dataset","VERIFIED"],
    ["2026-09-14","B","AM","317","17","35","working dataset","VERIFIED"],
    ["2026-09-11","B","AM","690","90","17","working dataset","VERIFIED"],
    ["2026-09-10","B","AM","545","45","44","working dataset","VERIFIED"],
    ["2026-09-09","B","AM","429","29","40","working dataset","VERIFIED"],
    ["2026-09-08","B","AM","945","45","63","working dataset","VERIFIED"],
]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLUMNS]
    for c in COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["market"] = df["market"].str.upper()
    df["session"] = df["session"].str.upper()
    df["verified"] = df["verified"].str.upper()
    return df


def validate_df(df: pd.DataFrame):
    errors = []
    if df.empty:
        return ["ไม่มีข้อมูล"]
    for i, r in df.iterrows():
        row = i + 2
        if not r["date"] or r["date"] == "NaT":
            errors.append(f"แถว {row}: date ไม่ถูกต้อง")
        if r["market"] not in MARKETS:
            errors.append(f"แถว {row}: market ต้องเป็น A, B หรือ C")
        if r["session"] not in SESSIONS:
            errors.append(f"แถว {row}: session ต้องเป็น AM หรือ PM")
        if r["three_digit"] and (len(r["three_digit"]) != 3 or not r["three_digit"].isdigit()):
            errors.append(f"แถว {row}: three_digit ต้องเป็นตัวเลข 3 หลัก")
        for col in ["two_digit_top", "two_digit_bottom"]:
            v = r[col]
            if v and (len(v) != 2 or not v.isdigit()):
                errors.append(f"แถว {row}: {col} ต้องเป็นตัวเลข 2 หลัก หรือว่าง")
    return errors


def digit_frequency(series):
    counts = {str(i): 0 for i in range(10)}
    for value in series.dropna().astype(str):
        for ch in value:
            if ch.isdigit():
                counts[ch] += 1
    return pd.DataFrame({"เลข": list(counts.keys()), "จำนวน": list(counts.values())})


if "dataset" not in st.session_state:
    st.session_state.dataset = normalize_df(pd.DataFrame(INITIAL_ROWS, columns=COLUMNS))

st.title("🎯 Stock Lottery Analyzer v1")
st.caption("Analyzer สำหรับ A/B/C × AM/PM — เน้นการจัดเก็บและวิเคราะห์ข้อมูลย้อนหลัง ไม่ใช่การรับประกันผลลัพธ์")

# -----------------------------
# Sidebar: six-market selection
# -----------------------------
with st.sidebar:
    st.header("⚙️ ตัวกรอง")
    market = st.selectbox("ตลาด", list(MARKETS.keys()), format_func=lambda x: f"{x} — {MARKETS[x]}")
    session = st.selectbox("รอบ", list(SESSIONS.keys()), format_func=lambda x: f"{x} — {SESSIONS[x]}")
    window = st.selectbox("ช่วงย้อนหลัง", [10, 20, 50, 100], index=0)
    st.divider()
    st.write("**6 กลุ่มที่รองรับ**")
    for m in MARKETS:
        st.write(f"• {m}-AM {MARKETS[m]}เช้า")
        st.write(f"• {m}-PM {MARKETS[m]}บ่าย")

# -----------------------------
# Upload / append CSV
# -----------------------------
st.subheader("📥 เพิ่มข้อมูลย้อนหลัง")

uploaded = st.file_uploader(
    "อัปโหลด CSV เพื่อเพิ่มข้อมูล (ใช้คอลัมน์ตามตัวอย่างด้านล่าง)",
    type=["csv"],
    accept_multiple_files=False,
)

if uploaded is not None:
    try:
        incoming = pd.read_csv(uploaded, dtype=str).fillna("")
        incoming = normalize_df(incoming)
        errs = validate_df(incoming)
        if errs:
            st.error("CSV มีข้อผิดพลาดบางรายการ")
            st.write(errs[:20])
        else:
            combined = pd.concat([st.session_state.dataset, incoming], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=["date","market","session"], keep="last")
            removed = before - len(combined)
            st.session_state.dataset = normalize_df(combined)
            st.success(f"เพิ่มข้อมูลแล้ว {len(incoming)} แถว • ตัดรายการซ้ำ {removed} แถว")
    except Exception as e:
        st.error(f"อ่าน CSV ไม่สำเร็จ: {e}")

# -----------------------------
# Current filtered data
# -----------------------------
df = st.session_state.dataset.copy()
filtered = df[(df["market"] == market) & (df["session"] == session)].copy()
filtered = filtered.sort_values("date", ascending=False)
recent = filtered.head(window)

# -----------------------------
# Summary cards
# -----------------------------
st.subheader(f"📊 {market} — {MARKETS[market]} / {session} — {SESSIONS[session]}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("ข้อมูลทั้งหมด", len(filtered))
c2.metric(f"ย้อนหลัง {window}", len(recent))
c3.metric("VERIFIED", int((filtered["verified"] == "VERIFIED").sum()))
c4.metric("ตลาดรวม", len(df))

# -----------------------------
# Frequency analysis
# -----------------------------
if recent.empty:
    st.info("ยังไม่มีข้อมูลของตลาด/รอบนี้ — สามารถเติมข้อมูลภายหลังได้")
else:
    st.subheader("🔢 วิเคราะห์ความถี่เลข")
    tab1, tab2, tab3 = st.tabs(["3 ตัวบน", "2 ตัวบน", "2 ตัวล่าง"])
    with tab1:
        st.dataframe(digit_frequency(recent["three_digit"]), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(digit_frequency(recent["two_digit_top"]), use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(digit_frequency(recent["two_digit_bottom"]), use_container_width=True, hide_index=True)

    st.subheader("📋 ผลย้อนหลัง")
    display_cols = ["date", "three_digit", "two_digit_top", "two_digit_bottom", "verified"]
    st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

# -----------------------------
# All-market status
# -----------------------------
st.subheader("🗂️ สถานะข้อมูล 6 ตลาด")
status_rows = []
for m in MARKETS:
    for s in SESSIONS:
        x = df[(df["market"] == m) & (df["session"] == s)]
        status_rows.append({
            "กลุ่ม": f"{m}-{s}",
            "ตลาด": MARKETS[m],
            "รอบ": SESSIONS[s],
            "จำนวนงวด": len(x),
            "VERIFIED": int((x["verified"] == "VERIFIED").sum()),
            "ล่าสุด": x["date"].max() if not x.empty else "-",
        })
st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

# -----------------------------
# Data manager
# -----------------------------
st.subheader("🛠️ จัดการข้อมูล")
with st.expander("ดู/แก้ไขข้อมูลทั้งหมด"):
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="data_editor",
    )
    if st.button("💾 บันทึกข้อมูลที่แก้ไข"):
        edited = normalize_df(edited)
        errs = validate_df(edited)
        if errs:
            st.error("บันทึกไม่ได้ เพราะมีข้อมูลไม่ถูกต้อง")
            st.write(errs[:20])
        else:
            st.session_state.dataset = edited.drop_duplicates(
                subset=["date","market","session"], keep="last"
            ).sort_values("date", ascending=False)
            st.success("บันทึกข้อมูลใน session แล้ว")

# -----------------------------
# Export
# -----------------------------
export_df = normalize_df(st.session_state.dataset).sort_values("date", ascending=False)
csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "⬇️ ดาวน์โหลด Dataset CSV",
    data=csv_bytes,
    file_name="stock_lottery_v1_dataset.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption("หมายเหตุ: ค่า 2 ตัวบนจะเก็บเฉพาะเมื่อมีข้อมูลระบุโดยแหล่งข้อมูล ไม่เดาจาก 3 ตัวบน")

import os
import io
import re
import datetime as dt
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

APP_TITLE = "Spend Tracker"
DB_PATH = os.getenv("SPENDTRACKER_DB", "spend_tracker.db")

# Secrets / env
APP_PIN = None
DATABASE_URL = None
try:
    if hasattr(st, "secrets"):
        if "APP_PIN" in st.secrets:
            APP_PIN = str(st.secrets["APP_PIN"])
        if "DATABASE_URL" in st.secrets:
            DATABASE_URL = str(st.secrets["DATABASE_URL"])
except Exception:
    pass

if APP_PIN is None:
    APP_PIN = os.getenv("APP_PIN")

if DATABASE_URL is None:
    DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("SPENDTRACKER_DATABASE_URL")

DEFAULT_CATEGORIES = [
    "Food & Drinks", "Transport", "Groceries", "Bills", "Rent", "Shopping",
    "Health", "Entertainment", "Education", "Gifts", "Subscriptions",
    "Travel", "Fees", "Other"
]
DEFAULT_ACCOUNTS = ["Cash", "Bank", "Wallet"]
DEFAULT_METHODS = ["Cash", "Card", "Transfer", "Mobile Wallet", "Other"]
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "EGP")

st.set_page_config(page_title=APP_TITLE, page_icon="💳", layout="wide")



def normalize_db_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    # Force SSL for hosted Postgres (Supabase expects SSL)
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url

def get_engine() -> Engine:
    """
    - If DATABASE_URL is set: use Postgres (recommended for hosted use).
    - Else: fallback to local SQLite file.
    """
    if DATABASE_URL:
        url = normalize_db_url(DATABASE_URL.strip())
        # Allow plain postgresql://
        if url.startswith("postgresql://") and "+psycopg" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return create_engine(url, pool_pre_ping=True)
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db(engine: Engine) -> None:
    idx = "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions (tdate);"
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_created TEXT NOT NULL,
                    ts_updated TEXT NOT NULL,
                    tdate TEXT NOT NULL,
                    ttype TEXT NOT NULL CHECK (ttype IN ('expense','income','transfer')),
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    category TEXT,
                    merchant TEXT,
                    account TEXT,
                    method TEXT,
                    note TEXT,
                    tags TEXT
                );
            """))
            conn.execute(text(idx))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id BIGSERIAL PRIMARY KEY,
                    ts_created TEXT NOT NULL,
                    ts_updated TEXT NOT NULL,
                    tdate DATE NOT NULL,
                    ttype TEXT NOT NULL CHECK (ttype IN ('expense','income','transfer')),
                    amount DOUBLE PRECISION NOT NULL,
                    currency TEXT NOT NULL,
                    category TEXT,
                    merchant TEXT,
                    account TEXT,
                    method TEXT,
                    note TEXT,
                    tags TEXT
                );
            """))
            conn.execute(text(idx))


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require_pin() -> None:
    if not APP_PIN:
        return
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return

    st.markdown("### 🔒 Enter PIN")
    pin = st.text_input("PIN", type="password")
    if st.button("Unlock"):
        if pin == str(APP_PIN):
            st.session_state.authed = True
            st.success("Unlocked.")
            st.rerun()
        else:
            st.error("Wrong PIN.")
    st.stop()


def upsert_transaction(engine: Engine, payload: Dict[str, Any], txn_id: Optional[int] = None) -> int:
    for k in ["tdate", "ttype", "amount", "currency"]:
        if payload.get(k) in (None, ""):
            raise ValueError(f"Missing required field: {k}")

    tdate = payload["tdate"]
    if isinstance(tdate, dt.date):
        tdate_str = tdate.isoformat()
    else:
        tdate_str = str(tdate)

    ttype = str(payload["ttype"]).lower().strip()
    if ttype not in ("expense", "income", "transfer"):
        raise ValueError("Type must be expense, income, or transfer.")

    amount = float(payload["amount"])
    if amount <= 0:
        raise ValueError("Amount must be > 0.")

    currency = str(payload["currency"]).strip().upper()[:10]
    category = (payload.get("category") or "").strip()[:100] or None
    merchant = (payload.get("merchant") or "").strip()[:200] or None
    account = (payload.get("account") or "").strip()[:100] or None
    method = (payload.get("method") or "").strip()[:100] or None
    note = (payload.get("note") or "").strip()[:1000] or None
    tags = (payload.get("tags") or "").strip()[:300] or None
    ts = now_iso()

    with engine.begin() as conn:
        if txn_id is None:
            if engine.dialect.name == "sqlite":
                conn.execute(text("""
                    INSERT INTO transactions
                    (ts_created, ts_updated, tdate, ttype, amount, currency, category, merchant, account, method, note, tags)
                    VALUES (:ts, :ts, :tdate, :ttype, :amount, :currency, :category, :merchant, :account, :method, :note, :tags)
                """), dict(ts=ts, tdate=tdate_str, ttype=ttype, amount=amount, currency=currency,
                           category=category, merchant=merchant, account=account, method=method, note=note, tags=tags))
                rid = conn.execute(text("SELECT last_insert_rowid()")).scalar_one()
                return int(rid)
            else:
                rid = conn.execute(text("""
                    INSERT INTO transactions
                    (ts_created, ts_updated, tdate, ttype, amount, currency, category, merchant, account, method, note, tags)
                    VALUES (:ts, :ts, :tdate, :ttype, :amount, :currency, :category, :merchant, :account, :method, :note, :tags)
                    RETURNING id
                """), dict(ts=ts, tdate=tdate_str, ttype=ttype, amount=amount, currency=currency,
                           category=category, merchant=merchant, account=account, method=method, note=note, tags=tags)).scalar_one()
                return int(rid)
        else:
            conn.execute(text("""
                UPDATE transactions SET
                  ts_updated=:ts,
                  tdate=:tdate,
                  ttype=:ttype,
                  amount=:amount,
                  currency=:currency,
                  category=:category,
                  merchant=:merchant,
                  account=:account,
                  method=:method,
                  note=:note,
                  tags=:tags
                WHERE id=:id
            """), dict(ts=ts, tdate=tdate_str, ttype=ttype, amount=amount, currency=currency,
                       category=category, merchant=merchant, account=account, method=method, note=note, tags=tags, id=int(txn_id)))
            return int(txn_id)


def delete_transaction(engine: Engine, txn_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM transactions WHERE id=:id"), {"id": int(txn_id)})


def read_transactions(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY tdate DESC, id DESC", conn)
    if not df.empty:
        df["tdate"] = pd.to_datetime(df["tdate"]).dt.date
        df["amount"] = df["amount"].astype(float)
    return df


def apply_filters(df: pd.DataFrame, date_from, date_to, ttypes, categories, accounts, q: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if date_from:
        out = out[out["tdate"] >= date_from]
    if date_to:
        out = out[out["tdate"] <= date_to]
    if ttypes and len(ttypes) < 3:
        out = out[out["ttype"].isin(ttypes)]
    if categories:
        out = out[out["category"].fillna("").isin(categories)]
    if accounts:
        out = out[out["account"].fillna("").isin(accounts)]
    if q and q.strip():
        qq = q.strip().lower()
        cols = ["merchant", "note", "tags", "category", "account", "method", "currency", "ttype"]
        mask = False
        for c in cols:
            mask = mask | out[c].fillna("").astype(str).str.lower().str.contains(qq, regex=False)
        out = out[mask]
    return out


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"net": 0.0, "income": 0.0, "expense": 0.0, "transfer": 0.0}
    income = float(df.loc[df["ttype"] == "income", "amount"].sum())
    expense = float(df.loc[df["ttype"] == "expense", "amount"].sum())
    transfer = float(df.loc[df["ttype"] == "transfer", "amount"].sum())
    net = income - expense
    return {"net": net, "income": income, "expense": expense, "transfer": transfer}


def to_excel_bytes(transactions: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        tx = transactions.copy()
        if not tx.empty:
            tx["tdate"] = pd.to_datetime(tx["tdate"]).dt.date
        tx.to_excel(writer, index=False, sheet_name="Transactions")

        if not tx.empty:
            tx2 = tx.copy()
            tx2["month"] = pd.to_datetime(tx2["tdate"]).dt.to_period("M").astype(str)
            monthly = tx2.pivot_table(index="month", columns="ttype", values="amount", aggfunc="sum", fill_value=0.0).reset_index()
            monthly["net_income_minus_expense"] = monthly.get("income", 0.0) - monthly.get("expense", 0.0)
            monthly.to_excel(writer, index=False, sheet_name="Monthly Summary")

            cat = (
                tx2[tx2["ttype"] == "expense"]
                .groupby(tx2["category"].fillna("Uncategorized"))["amount"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"amount": "expense_total"})
            )
            cat.to_excel(writer, index=False, sheet_name="Expense by Category")

    return output.getvalue()


def detect_duplicates(df: pd.DataFrame, payload: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.head(0)
    try:
        tdate = payload["tdate"]
        if not isinstance(tdate, dt.date):
            tdate = pd.to_datetime(tdate).date()
        ttype = str(payload["ttype"])
        amount = float(payload["amount"])
        merchant = (payload.get("merchant") or "").strip().lower()
        cand = df[(df["tdate"] == tdate) & (df["ttype"] == ttype) & (df["amount"] == amount)]
        if merchant:
            cand = cand[cand["merchant"].fillna("").str.lower().str.contains(merchant, regex=False)]
        return cand
    except Exception:
        return df.head(0)


def csv_import(df_csv: pd.DataFrame) -> pd.DataFrame:
    if df_csv is None or df_csv.empty:
        return pd.DataFrame()

    cols = {c.lower().strip(): c for c in df_csv.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    map_date = pick("tdate", "date", "transaction_date", "posted_date")
    map_type = pick("ttype", "type", "transaction_type")
    map_amt = pick("amount", "amt", "value", "debit", "credit")
    map_cur = pick("currency", "cur")
    map_cat = pick("category", "cat")
    map_mer = pick("merchant", "payee", "description", "details")
    map_acc = pick("account", "acct")
    map_meth = pick("method", "payment_method")
    map_note = pick("note", "notes", "memo")
    map_tags = pick("tags", "tag")

    out = pd.DataFrame()
    out["tdate"] = pd.to_datetime(df_csv[map_date]) if map_date else pd.NaT

    if map_type:
        out["ttype"] = df_csv[map_type].astype(str).str.lower().str.strip()
    else:
        out["ttype"] = "expense"

    if map_amt and map_amt in df_csv.columns:
        out["amount"] = pd.to_numeric(df_csv[map_amt], errors="coerce").abs()
    else:
        debit_col = pick("debit")
        credit_col = pick("credit")
        if debit_col and credit_col:
            debit = pd.to_numeric(df_csv[debit_col], errors="coerce").fillna(0)
            credit = pd.to_numeric(df_csv[credit_col], errors="coerce").fillna(0)
            out["amount"] = (debit.abs() + credit.abs())
            out.loc[credit > 0, "ttype"] = "income"
            out.loc[debit > 0, "ttype"] = "expense"
        else:
            out["amount"] = pd.NA

    out["currency"] = (df_csv[map_cur] if map_cur else DEFAULT_CURRENCY).astype(str).str.upper()
    out["category"] = (df_csv[map_cat] if map_cat else "").astype(str)
    out["merchant"] = (df_csv[map_mer] if map_mer else "").astype(str)
    out["account"] = (df_csv[map_acc] if map_acc else "").astype(str)
    out["method"] = (df_csv[map_meth] if map_meth else "").astype(str)
    out["note"] = (df_csv[map_note] if map_note else "").astype(str)
    out["tags"] = (df_csv[map_tags] if map_tags else "").astype(str)

    out = out.dropna(subset=["tdate", "amount"])
    out["tdate"] = pd.to_datetime(out["tdate"]).dt.date
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[out["amount"] > 0]
    out["ttype"] = out["ttype"].where(out["ttype"].isin(["expense", "income", "transfer"]), "expense")
    out["currency"] = out["currency"].fillna(DEFAULT_CURRENCY).astype(str).str.upper().str.slice(0, 10)
    for c in ["category", "merchant", "account", "method", "note", "tags"]:
        out[c] = out[c].fillna("").astype(str).str.strip()
    return out


def parse_quick_capture(text_line: str) -> Dict[str, Any]:
    """
    Voice-typing parser. Example:
    "Expense 370 EGP category shopping merchant plastic dish account cash method cash note for my room"

    Keywords: expense|income|transfer, category, merchant, account, method, tags, note, date (YYYY-MM-DD or YYYY/MM/DD)
    """
    if not text_line or not text_line.strip():
        return {}

    raw = re.sub(r"\s+", " ", text_line.strip())
    lower = raw.lower()

    # date
    date_match = re.search(r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b", raw)
    if date_match:
        ds = date_match.group(1).replace("/", "-")
        try:
            parsed_date = pd.to_datetime(ds).date()
        except Exception:
            parsed_date = dt.date.today()
    else:
        parsed_date = dt.date.today()

    # type
    ttype = "expense"
    for cand in ["expense", "income", "transfer"]:
        if re.search(rf"\b{cand}\b", lower):
            ttype = cand
            break

    # amount: first number
    amt_match = re.search(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)", raw)
    amount = float(amt_match.group(1)) if amt_match else None

    # currency: first 3-4 letters token that isn't a keyword
    currency = DEFAULT_CURRENCY
    keywords = {"expense","income","transfer","category","merchant","account","method","tags","tag","note","date"}
    for tok in raw.split():
        tl = tok.lower().strip(",.:;")
        if tl in keywords:
            continue
        if re.fullmatch(r"[A-Za-z]{3,4}", tok):
            currency = tok.upper()
            break

    # segment extraction
    keys = ["category", "merchant", "account", "method", "tags", "tag", "note", "date"]
    positions = []
    for kw in keys:
        m = re.search(rf"\b{kw}\b", lower)
        if m:
            positions.append((m.start(), kw))
    positions.sort()

    def seg(kw: str) -> str:
        for i, (pos, k) in enumerate(positions):
            if k == kw:
                end = len(raw)
                if i + 1 < len(positions):
                    end = positions[i + 1][0]
                chunk = raw[pos:end]
                chunk = re.sub(rf"(?i)^\s*{kw}\s*", "", chunk).strip()
                return chunk.strip(" ,.:;")
        return ""

    category = seg("category")
    merchant = seg("merchant")
    account = seg("account")
    method = seg("method")
    tags = seg("tags") or seg("tag")
    note = seg("note")

    return {
        "tdate": parsed_date,
        "ttype": ttype,
        "amount": amount,
        "currency": currency,
        "category": category or "",
        "merchant": merchant or "",
        "account": account or "",
        "method": method or "",
        "tags": tags or "",
        "note": note or "",
        "raw": raw
    }


def main():
    require_pin()

    st.title("💳 Spend Tracker (Phone-friendly)")
    st.caption("Log expenses/income fast, then export a clean Excel file.")

    engine = get_engine()
    init_db(engine)

    with st.sidebar:
        st.header("⚙️ Setup")
        categories = st.multiselect("Categories (for dropdown)", DEFAULT_CATEGORIES, default=DEFAULT_CATEGORIES)
        accounts = st.multiselect("Accounts", DEFAULT_ACCOUNTS, default=DEFAULT_ACCOUNTS)
        methods = st.multiselect("Payment methods", DEFAULT_METHODS, default=DEFAULT_METHODS)
        st.divider()
        st.subheader("🔎 Filters")
        today = dt.date.today()
        date_from = st.date_input("From", value=today.replace(day=1))
        date_to = st.date_input("To", value=today)
        ttypes = st.multiselect("Types", ["expense", "income", "transfer"], default=["expense", "income", "transfer"])
        f_categories = st.multiselect("Category", [""] + sorted(set(categories)), default=[])
        f_accounts = st.multiselect("Account", [""] + sorted(set(accounts)), default=[])
        q = st.text_input("Search", placeholder="merchant, note, tag…")

    df_all = read_transactions(engine)
    df = apply_filters(df_all, date_from, date_to, ttypes, f_categories, f_accounts, q)

    s = summarize(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net (Income - Expense)", f"{s['net']:,.2f}")
    c2.metric("Income", f"{s['income']:,.2f}")
    c3.metric("Expense", f"{s['expense']:,.2f}")
    c4.metric("Transfers", f"{s['transfer']:,.2f}")

    st.divider()
    tab_add, tab_view, tab_import, tab_export = st.tabs(["➕ Add", "📒 View/Edit", "⬆️ Import CSV", "📤 Export Excel"])

    with tab_add:
        st.subheader("Add a transaction")

        st.markdown("#### 🎙️ Quick capture (voice typing)")
        st.caption("On your phone: tap the keyboard mic, speak a sentence, then tap **Parse** to fill the form.")
        qc = st.text_input(
            "Quick capture sentence",
            placeholder="Expense 370 EGP category shopping merchant plastic dish account cash method cash note for my room",
            key="quick_capture_text"
        )
        qc_col1, qc_col2 = st.columns([1, 3])
        with qc_col1:
            if st.button("Parse", key="quick_capture_parse"):
                parsed = parse_quick_capture(qc)
                if not parsed:
                    st.warning("Type a sentence first.")
                else:
                    st.session_state["add_date"] = parsed["tdate"]
                    st.session_state["add_type"] = parsed["ttype"]
                    if parsed["amount"] is not None:
                        st.session_state["add_amount"] = float(parsed["amount"])
                    st.session_state["add_currency"] = parsed["currency"]
                    st.session_state["add_category"] = parsed["category"]
                    st.session_state["add_merchant"] = parsed["merchant"]
                    st.session_state["add_account"] = parsed["account"]
                    st.session_state["add_method"] = parsed["method"]
                    st.session_state["add_tags"] = parsed["tags"]
                    if parsed["note"].strip():
                        st.session_state["add_note"] = parsed["note"]
                    st.success("Parsed. Review fields below, then Save.")
        with qc_col2:
            st.caption("Keywords: **expense/income/transfer**, **category**, **merchant**, **account**, **method**, **tags**, **note**, optional date like **2025-12-20**.")
        st.divider()

        colL, colR = st.columns([1, 1])
        with colL:
            tdate = st.date_input("Date", value=dt.date.today(), key="add_date")
            ttype = st.selectbox("Type", ["expense", "income", "transfer"], index=0, key="add_type")
            amount = st.number_input("Amount", min_value=0.0, value=0.0, step=10.0, format="%.2f", key="add_amount")
            currency = st.text_input("Currency", value=DEFAULT_CURRENCY, key="add_currency").upper()
            category = st.selectbox("Category", [""] + list(categories), index=0, key="add_category")
        with colR:
            merchant = st.text_input("Merchant / Payee", value="", key="add_merchant")
            account = st.selectbox("Account", [""] + list(accounts), index=0, key="add_account")
            method = st.selectbox("Payment method", [""] + list(methods), index=0, key="add_method")
            tags = st.text_input("Tags (comma-separated)", value="", key="add_tags")
            note = st.text_area("Note", value="", height=90, key="add_note")

        payload = {
            "tdate": tdate, "ttype": ttype, "amount": amount, "currency": currency,
            "category": category, "merchant": merchant, "account": account,
            "method": method, "note": note, "tags": tags
        }

        dups = detect_duplicates(df_all, payload)
        if not dups.empty:
            st.warning("Possible duplicate(s) detected (same date/type/amount):")
            st.dataframe(dups[["id", "tdate", "ttype", "amount", "merchant", "category"]], use_container_width=True)

        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("Save", key="add_save"):
                try:
                    new_id = upsert_transaction(engine, payload)
                    st.success(f"Saved. (ID {new_id})")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with colB:
            st.caption("Tip: use **Transfer** for moving money between accounts so it doesn't distort spending.")

    with tab_view:
        st.subheader("Transactions")
        if df.empty:
            st.info("No transactions in this range.")
        else:
            show_cols = ["id", "tdate", "ttype", "amount", "currency", "category", "merchant", "account", "method", "tags", "note", "ts_updated"]
            st.dataframe(df[show_cols], use_container_width=True, height=420)

            st.markdown("#### Edit / Delete")
            default_id = int(df.iloc[0]["id"]) if not df.empty else 1
            edit_id = st.number_input("Transaction ID", min_value=1, step=1, value=default_id, key="edit_id")

            row = df_all[df_all["id"] == int(edit_id)]
            if row.empty:
                st.warning("ID not found.")
            else:
                r = row.iloc[0]
                e1, e2 = st.columns([1, 1])
                with e1:
                    etdate = st.date_input("Date", value=r["tdate"], key="edit_date")
                    ettype = st.selectbox(
                        "Type",
                        ["expense", "income", "transfer"],
                        index=["expense", "income", "transfer"].index(r["ttype"]),
                        key="edit_type"
                    )
                    eamount = st.number_input("Amount", min_value=0.0, value=float(r["amount"]), step=10.0, format="%.2f", key="edit_amount")
                    ecurrency = st.text_input("Currency", value=str(r["currency"]), key="edit_currency").upper()
                    ecategory = st.selectbox(
                        "Category",
                        [""] + list(categories),
                        index=([""] + list(categories)).index(r["category"] or "") if (r["category"] or "") in ([""] + list(categories)) else 0,
                        key="edit_category"
                    )
                with e2:
                    emerchant = st.text_input("Merchant / Payee", value=str(r["merchant"] or ""), key="edit_merchant")
                    eaccount = st.selectbox(
                        "Account",
                        [""] + list(accounts),
                        index=([""] + list(accounts)).index(r["account"] or "") if (r["account"] or "") in ([""] + list(accounts)) else 0,
                        key="edit_account"
                    )
                    emethod = st.selectbox(
                        "Payment method",
                        [""] + list(methods),
                        index=([""] + list(methods)).index(r["method"] or "") if (r["method"] or "") in ([""] + list(methods)) else 0,
                        key="edit_method"
                    )
                    etags = st.text_input("Tags", value=str(r["tags"] or ""), key="edit_tags")
                    enote = st.text_area("Note", value=str(r["note"] or ""), height=90, key="edit_note")

                upd = {
                    "tdate": etdate, "ttype": ettype, "amount": eamount, "currency": ecurrency,
                    "category": ecategory, "merchant": emerchant, "account": eaccount,
                    "method": emethod, "note": enote, "tags": etags
                }

                b1, b2, b3 = st.columns([1, 1, 2])
                with b1:
                    if st.button("Update", key="edit_update"):
                        try:
                            upsert_transaction(engine, upd, txn_id=int(edit_id))
                            st.success("Updated.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                with b2:
                    if st.button("Delete", type="secondary", key="edit_delete"):
                        delete_transaction(engine, int(edit_id))
                        st.success("Deleted.")
                        st.rerun()
                with b3:
                    st.caption("Delete is permanent in this version (keeps it simple).")

    with tab_import:
        st.subheader("Import CSV (bank statement / any CSV)")
        st.caption("Upload a CSV and we’ll map common column names automatically. Preview before importing.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
        if uploaded is not None:
            try:
                df_csv = pd.read_csv(uploaded)
                st.write("CSV preview:")
                st.dataframe(df_csv.head(20), use_container_width=True)

                mapped = csv_import(df_csv)
                st.write("Mapped transactions preview:")
                st.dataframe(mapped.head(20), use_container_width=True)

                if mapped.empty:
                    st.warning("Could not map any rows. Make sure your CSV has date and amount columns.")
                else:
                    if st.button(f"Import {len(mapped):,} rows", key="csv_import_btn"):
                        ok = 0
                        fail = 0
                        for _, rr in mapped.iterrows():
                            try:
                                upsert_transaction(engine, {
                                    "tdate": rr["tdate"],
                                    "ttype": rr["ttype"],
                                    "amount": float(rr["amount"]),
                                    "currency": rr.get("currency", DEFAULT_CURRENCY),
                                    "category": rr.get("category", ""),
                                    "merchant": rr.get("merchant", ""),
                                    "account": rr.get("account", ""),
                                    "method": rr.get("method", ""),
                                    "note": rr.get("note", ""),
                                    "tags": rr.get("tags", ""),
                                })
                                ok += 1
                            except Exception:
                                fail += 1
                        st.success(f"Imported {ok} rows. Failed {fail} rows.")
                        st.rerun()
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    with tab_export:
        st.subheader("Export to Excel")
        st.caption("Exports your current filtered view + summary sheets.")
        if df.empty:
            st.info("Nothing to export in the current filter range.")
        else:
            excel_bytes = to_excel_bytes(df)
            fname = f"spend_tracker_{dt.date.today().isoformat()}.xlsx"
            st.download_button(
                label="Download Excel (.xlsx)",
                data=excel_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.divider()
    with st.expander("🛠️ Advanced (Backups)"):
        if engine.dialect.name != "sqlite":
            st.info("You are using a cloud database (Postgres). For backups: use your provider dashboard, and/or export Excel from the Export tab.")
        else:
            st.caption("SQLite mode: download the DB file for a full backup.")
            if os.path.exists(DB_PATH):
                with open(DB_PATH, "rb") as f:
                    st.download_button("Download DB backup (.db)", f.read(), file_name="spend_tracker.db", use_container_width=True)


if __name__ == "__main__":
    main()

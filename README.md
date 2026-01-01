# Spend Tracker (Phone-friendly) 💳

## Goal
Use it **regularly on your phone** (like an app), and keep your data safe even if the app restarts.

## Best free setup (recommended)
- **Host the app** on Streamlit Community Cloud (free)
- **Store data** in Supabase Postgres (free)

### Why Supabase?
Supabase Free includes **500 MB database**, and free projects pause after **1 week of inactivity**. If you use it regularly, it stays active.

---

## 1) Create Supabase database (Free)
1. Create a Supabase account
2. Create a **New project**
3. In the dashboard, click **Connect** and copy the **Pooler** connection string (Session or Transaction mode).
   - This avoids IPv4/IPv6 headaches.

You will get a URL that looks like:
`postgresql://USER:PASSWORD@HOST:PORT/postgres`

---

## 2) Put the code on GitHub
Create a repo (e.g. `spend-tracker`) and upload:
- `app.py`
- `requirements.txt`
- `README.md`

---

## 3) Deploy on Streamlit Community Cloud (Free)
1. Streamlit Cloud → **New app**
2. Choose your repo
3. Main file path: `app.py`
4. Deploy

---

## 4) Add Secrets (VERY IMPORTANT)
In Streamlit Cloud → App → Settings → **Secrets**, add:

```toml
DATABASE_URL="PASTE_SUPABASE_POOLER_URL_HERE"
APP_PIN="1234"  # optional
```

---

## 5) Use it like an app on your phone
- Android (Chrome): ⋮ → **Add to Home screen**
- iPhone (Safari): Share → **Add to Home Screen**

---

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

# Reproducibility Guide — Step by Step

This guide assumes zero prior setup. Pick the path that matches where you
want to work: VS Code (local machine), Google Colab (no local install), or
"I just want to push this to GitHub."

---

## Path A — VS Code (local machine, Windows/Mac/Linux)

### A1. Install prerequisites (one-time)
1. Install Python 3.10+ from https://www.python.org/downloads/ (tick "Add
   Python to PATH" on Windows).
2. Install VS Code from https://code.visualstudio.com/.
3. In VS Code, install the "Python" extension (Ctrl+Shift+X, search
   "Python", install the Microsoft one).
4. Install Git from https://git-scm.com/downloads.

### A2. Get the project onto your machine
Open VS Code's integrated terminal (`` Ctrl+` ``) and run:
```bash
git clone https://github.com/dhamangeraashi-bit/xai-credit-risk-assessment.git
cd xai-credit-risk-assessment
```
(If you haven't pushed to GitHub yet, see Path C first, or just unzip the
project folder you were given and `cd` into it instead.)

### A3. Create a virtual environment (recommended, keeps things clean)
```bash
python -m venv venv
```
Activate it:
- Windows (PowerShell): `venv\Scripts\Activate.ps1`
- Windows (cmd.exe): `venv\Scripts\activate.bat`
- Mac/Linux: `source venv/bin/activate`

You should now see `(venv)` at the start of your terminal prompt.

### A4. Install dependencies
```bash
pip install -r requirements.txt
```

### A5. In VS Code, select the right interpreter
`Ctrl+Shift+P` -> "Python: Select Interpreter" -> pick the one inside
`venv/` (it will say something like `('venv': venv)`).

### A6. Run the whole pipeline
```bash
python scripts/run_all.py
```
Watch the terminal — each script prints its own log lines. When it says
"All scripts completed successfully," open the `outputs/figures/` folder
in VS Code's file explorer (left sidebar) and click any `.png` to preview it.

### A7. Open the dashboard
Right-click `dashboard/dashboard.html` in VS Code's file explorer ->
"Reveal in File Explorer" (or Finder on Mac) -> double-click it -> it opens
in your default browser. No server needed.

### A8. Run individual scripts (optional)
If you only want to regenerate one figure, e.g. the ROC curves:
```bash
python scripts/02_train_models.py   # must run first, always
python scripts/03_generate_roc_curves.py
```

---

## Path B — Google Colab (no local install needed)

### B1. Open a new notebook
Go to https://colab.research.google.com/ -> "New notebook."

### B2. Get the project into Colab
**Option 1 — clone from GitHub (if you've already pushed it, see Path C):**
```python
!git clone https://github.com/dhamangeraashi-bit/xai-credit-risk-assessment.git
%cd xai-credit-risk-assessment
```

**Option 2 — upload the zip you were given:**
```python
from google.colab import files
uploaded = files.upload()   # choose xai-credit-risk-assessment.zip in the dialog
!unzip -q xai-credit-risk-assessment.zip
%cd xai-credit-risk-assessment
```

### B3. Install dependencies
```python
!pip install -q -r requirements.txt
```

### B4. Run everything
```python
!python scripts/run_all.py
```

### B5. View a figure inline
```python
from IPython.display import Image
Image("outputs/figures/fig4_shap_summary.png")
```

### B6. View a table
```python
import pandas as pd
pd.read_csv("outputs/tables/table_II_model_comparison.csv")
```

### B7. Download all outputs to your computer
```python
!zip -r outputs.zip outputs/
from google.colab import files
files.download("outputs.zip")
```

### B8. View the dashboard in Colab
Colab can't open a local HTML file as a live page directly, but you can
render it inline:
```python
from IPython.display import IFrame
IFrame(src="dashboard/dashboard.html", width=1000, height=700)
```
(For the full interactive experience, download `dashboard/dashboard.html`
via the Colab file browser on the left and open it in a normal browser tab.)

---

## Path C — Pushing this project to your own GitHub repository

### C1. Create the repository on GitHub
1. Go to https://github.com/new
2. Repository name: `xai-credit-risk-assessment` (or your preferred name)
3. Keep it **public** (so the paper's "Code Availability" link works for
   reviewers), do **not** initialize with a README/license (you already have
   both locally)
4. Click "Create repository." Copy the URL it shows you, e.g.
   `https://github.com/<your-username>/xai-credit-risk-assessment.git`

### C2. Push your local project
From the project's root folder (wherever you unzipped it):
```bash
cd xai-credit-risk-assessment
git init
git add .
git commit -m "Initial commit: reproducible pipeline for XAI credit risk paper"
git branch -M main
git remote add origin https://github.com/<your-username>/xai-credit-risk-assessment.git
git push -u origin main
```
If prompted for credentials, GitHub no longer accepts your account password
directly — use a Personal Access Token instead:
GitHub -> Settings -> Developer settings -> Personal access tokens ->
Generate new token (classic) -> tick "repo" scope -> use this token as your
password when Git asks.

### C3. Verify
Refresh your GitHub repository page in the browser — you should see all
files (`README.md`, `src/`, `scripts/`, `data/raw/german.csv`,
`dashboard/dashboard.html`, etc.). The `outputs/` folder's contents from
your last local run will also be there unless you rely on `.gitignore`
(only `outputs/logs/*.log` and `data/processed/` are excluded by default —
figures and tables are tracked so reviewers can see them without running
anything).

### C4. Update the paper's "Code Availability" link
Make sure the manuscript's Code Availability section points at exactly this
URL: `https://github.com/<your-username>/xai-credit-risk-assessment`

### C5. Future updates
```bash
git add .
git commit -m "Describe what changed"
git push
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run scripts from the **project root** (`python scripts/02_train_models.py`), not from inside `scripts/` |
| `FileNotFoundError: pipeline_state.pkl` | You ran a later script before `02_train_models.py`. Run scripts in the order listed in `docs/FIGURE_MAPPING.md`, or just use `scripts/run_all.py` |
| `pip install` fails on an exact version | Drop the `==x.y.z` pin for that one package in `requirements.txt` and let pip pick a compatible newer version |
| LIME step (`07_shap_vs_lime.py`) is slow | Expected — LIME is ~200x slower than SHAP per the paper's own finding (Fig. 6). 40 applicants typically takes under a minute |
| Figures look different from the ones in this guide | Check you're on the pinned dependency versions in `requirements.txt`; minor library version differences can shift plot styling (not the underlying numbers) |
| Numbers differ slightly from the paper | Expected and documented — see `docs/ASSUMPTIONS.md` for exactly which numbers, why, and by how much |

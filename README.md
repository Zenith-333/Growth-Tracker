# Child Growth Status Checker

A Streamlit app that predicts a child's height-for-age (stunting) status and
weight-for-height (BMI) status from basic growth-monitoring inputs, using two
pre-trained scikit-learn models.

## Folder contents

```
app.py                          # the Streamlit app
requirements.txt                # pinned Python dependencies
models/
  feature_encoders.pkl          # LabelEncoders for Sex / Vaccinated / 4Ps
  hfa_model.pkl                 # LogisticRegression -> stunting status
  hfa_scaler.pkl                # StandardScaler for the HFA model
  hfa_label_encoder.pkl         # decodes HFA prediction back to a label
  bmi_model.pkl                 # DecisionTreeClassifier -> BMI status
  bmi_scaler.pkl                # StandardScaler for the BMI model
  bmi_label_encoder.pkl         # decodes BMI prediction back to a label
```

**Do not rename the files inside `models/`** — `app.py` loads them by these
exact names.

## 1. Put this on GitHub

If you don't already have a repo:

1. Go to https://github.com/new, give it a name (e.g. `child-growth-app`), keep it
   Public (Streamlit Cloud's free tier needs a public repo, or a private one
   linked to your Streamlit account), and click **Create repository**.
2. On the new repo's page, click **uploading an existing file**, then drag in:
   - `app.py`
   - `requirements.txt`
   - the whole `models/` folder (drag the folder in; GitHub's web uploader
     preserves the folder structure)
3. Commit the files.

Or, from your computer with git installed:

```bash
cd child-growth-app        # the folder containing app.py, requirements.txt, models/
git init
git add .
git commit -m "Initial commit: child growth status app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **New app**.
3. Pick your repository and branch (`main`), and set **Main file path** to
   `app.py`.
4. Click **Deploy**. The first build takes a minute or two while it installs
   `requirements.txt`.

Your app will be live at a URL like
`https://<your-app-name>.streamlit.app`. Any time you push new commits to the
repo, Streamlit Cloud redeploys automatically.

## Notes

- `scikit-learn` is pinned to `1.6.1` in `requirements.txt` because that's the
  version the models were trained with — using a different version can throw
  off predictions or raise unpickling errors.
- The models expect exactly these six inputs, in this order:
  `Age_in_Months, Sex, Weight_kg, Height_cm, Vaccinated, 4ps_Beneficiary`.
- This tool gives an automated estimate only and isn't a substitute for
  assessment by a qualified health worker.

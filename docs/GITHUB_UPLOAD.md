# Uploading the project to GitHub

This guide walks through publishing the repository so it satisfies the
project requirements: a tagged release, a clean README, runnable
one-command scripts, and no large data files committed.

> **Time required:** ~10 minutes
> **Prerequisites:** a GitHub account, `git` installed locally

---

## 1. Create the remote repository

1. Go to <https://github.com/new>.
2. Name it `turbofan-rul` (or anything you prefer — just remember to update
   the URL in the README).
3. Set visibility to **Public** so the graders can clone it.
4. **Do not** tick "Add a README", "Add .gitignore", or "Choose a license"
   — we have those locally already; ticking them would force a merge.
5. Click **Create repository**. GitHub shows a "quick setup" page — keep
   it open, you'll need the URL.

---

## 2. Initialise the local repository

Open a terminal in the project folder (the one that contains `Makefile`,
`README.md`, `src/`, etc.).

```bash
# 2.1 Verify nothing is committed yet
git status                          # should say "not a git repository"

# 2.2 Initialise + first commit
git init
git branch -M main
git add .
git commit -m "Initial commit: full pipeline + report v1.0"
```

If `git status` shows an unwanted file (e.g. `data/raw/*.txt` or
`models/*.pt`), confirm that `.gitignore` is in the project root and that
the file is matched. The committed `.gitignore` already excludes the raw
NASA files, large `*.pt` checkpoints, parquet caches, and LaTeX build
artifacts. Re-run `git add .` after fixing.

---

## 3. Push to GitHub

Replace `<USER>` with your GitHub username:

```bash
git remote add origin https://github.com/<USER>/turbofan-rul.git
git push -u origin main
```

If GitHub asks for authentication and you don't have SSH set up, use a
**personal access token** as the password. Generate one at
<https://github.com/settings/tokens> (classic, "repo" scope is sufficient).

---

## 4. Create the tagged release `v1.0`

The grading rubric specifically asks for a tagged release. Do this *after*
the main branch is pushed and you're happy with what's there.

```bash
git tag -a v1.0 -m "v1.0 — final submission for CSCI 447"
git push origin v1.0
```

Then on github.com:

1. Open your repo → **Releases** (right-hand sidebar) → **Draft a new
   release**.
2. **Choose a tag:** select `v1.0`.
3. **Release title:** `v1.0 — Final submission`.
4. **Description:** paste the headline-results table from the README and
   a one-paragraph summary.
5. Click **Publish release**.

The release URL becomes the canonical link to put in the report's Code
Availability section.

---

## 5. Verify that a fresh clone works end-to-end

This is the test the graders will perform. Do it yourself first.

**Linux / macOS:**

```bash
cd /tmp
git clone https://github.com/<USER>/turbofan-rul.git
cd turbofan-rul
make install        # install dependencies
make data           # download C-MAPSS
make pipeline-one SUB=FD001
make test           # 28 tests should pass
```

**Windows PowerShell:**

```powershell
cd $env:TEMP
git clone https://github.com/<USER>/turbofan-rul.git
cd turbofan-rul
.\run.ps1 install
.\run.ps1 data
.\run.ps1 pipeline-one FD001
.\run.ps1 test
```

If anything breaks, fix it on your local copy, commit, and push again. If
you re-tag `v1.0` you have to delete the old tag first:

```bash
git tag -d v1.0
git push --delete origin v1.0
git tag -a v1.0 -m "v1.0 — final submission for CSCI 447 (rev)"
git push origin v1.0
```

---

## 6. Extras worth doing (not required, but help readability)

- **Add a banner image.** Drop `docs/banner.png` and reference it at the
  top of the README with `![banner](docs/banner.png)`.
- **Pin the repo** on your GitHub profile so it's the first thing
  reviewers see.
- **Add topics.** On the repo page, click the gear icon next to "About"
  and add: `predictive-maintenance`, `xgboost`, `lstm`, `c-mapss`,
  `time-series`, `pytorch`. This helps discoverability.
- **GitHub Actions CI (optional).** Put the YAML below at
  `.github/workflows/ci.yml` to run the tests on every push:

  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.10" }
        - run: pip install -r requirements.txt
        - run: pytest tests/ -v
  ```

  A green badge next to the README is a credibility marker that grading
  panels appreciate.

---

## 7. Troubleshooting

| Symptom                                    | Likely cause                            | Fix                                                   |
| :---                                       | :---                                    | :---                                                  |
| `data/raw/train_FD001.txt` got committed   | `.gitignore` not in root before `git add` | `git rm --cached data/raw/*.txt && git commit`        |
| Push rejected: file too large              | a `*.pt` checkpoint slipped in           | same as above; verify `.gitignore` then re-add        |
| `make: command not found` (Windows)        | Windows ships without GNU make           | use `.\run.ps1 <task> <subset>` instead — same commands, native PowerShell |
| GitHub auth fails                          | token without `repo` scope               | regenerate the token at *Settings → Developer settings* |
| `pytest` can't import `config`             | running from the wrong directory         | run `pytest` from the project root, not from `tests/` |

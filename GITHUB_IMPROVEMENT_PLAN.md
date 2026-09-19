# GitHub Presence Improvement Plan

A practical, prioritized plan for making Shane Hurley's GitHub profile work as hard as his résumé does. Audience: engineering recruiters, ATS-driven résumé screeners, and hiring managers who click the `github.com/ShaneHurley` link. Guiding principle: **everything a visitor sees in the first 30 seconds should be truthful, polished, and point somewhere worth clicking.**

---

## 1. Profile README (highest leverage, ~30 minutes)

GitHub renders a special README at the top of your profile page if you create a repository named exactly after your username.

**Steps:**

1. Create a new **public** repository named `ShaneHurley` (i.e. `github.com/ShaneHurley/ShaneHurley`). GitHub will show a "✨ special repository" banner confirming it worked.
2. Initialize it with a `README.md`.
3. Paste in the contents of [`PROFILE_README.md`](PROFILE_README.md) from this repo (already written for this purpose — intro, featured BasketballElo section, coding journey, skills badges, leadership, contact links).
4. Verify the shields.io badges render (they load from `img.shields.io`, no setup needed).
5. Optional later: add a GitHub stats card (e.g. `github-readme-stats`) — nice-to-have, not essential, and some recruiters find them noisy. The current content-first layout is deliberate.

**Why it matters:** the profile page is the landing page for the résumé link. Today it shows default pinned-repo noise; after this step it tells a coherent story in one scroll.

---

## 2. Pin the right repositories

You can pin up to 6 repos on your profile. Recommended order:

1. **BasketballElo** — the flagship. Five-year arc, tests, docs site, leak registry.
2. **ShaneHurley** (the profile README repo) — optional; some people pin it, some leave it off since it's already the profile content.
3–6. Any cleaned-up older projects (see §6) — a polished early notebook repo is worth more pinned than a raw dump.

Avoid pinning forks you didn't meaningfully change or repos with no README.

---

## 3. Hygiene for THIS repo (BasketballElo)

This is the repo recruiters will actually open. Highest-impact fixes first:

### 3.1 Repo metadata (5 minutes, do first)
- **Description:** set the repo description (gear icon on the repo page → About) to something like: *"Leak-aware NBA spread prediction pipeline — Glicko-2 ratings, walk-forward evaluation, formal leak registry, FastAPI dashboard."*
- **Website:** set to `https://shanehurley.github.io/BasketballElo/`
- **Topics:** add `nba`, `sports-analytics`, `elo`, `glicko-2`, `machine-learning`, `python`, `walk-forward`, `calibration`, `sports-betting`, `data-pipeline`. Topics drive GitHub search/topic-page discovery.

### 3.2 Social preview image (15 minutes)
Repo **Settings → General → Social preview** — upload a 1280×640 image. This is what renders when the link is pasted into LinkedIn, Slack, iMessage, or a recruiter's ATS notes. A good option: a screenshot of the T-60 dashboard's calibration charts with the repo name overlaid. Without it, link unfurls show a generic avatar.

### 3.3 Releases and tags (30 minutes)
The repo has meaningful milestones (modularization, leak-free rewrite) but no GitHub Releases. Create at least one:
- Tag the current state as `v1.0.0` (`git tag -a v1.0.0 -m "Modular pipeline, leak registry, T-60 dashboard" && git push origin v1.0.0`), then draft a GitHub Release from the tag summarizing what's stable.
- Going forward, tag when roadmap epics land. Releases signal "maintained software," not "homework dump."

### 3.4 The 92 MB `allData.zip` in git history (1–2 hours, careful)
`allData.zip` (~88 MB compressed) is **tracked in git**, which bloats every clone. Options, in order of safety:

- **Option A (recommended, low risk):** leave history alone, but make sure the data story is documented — the README already notes raw data isn't shipped. Add `allData.zip` to `.gitignore` only if you intend to stop tracking it going forward, and consider moving data distribution to a GitHub Release asset (releases host files up to 2 GB without polluting clones).
- **Option B (thorough, higher risk):** purge it from history with [git-filter-repo](https://github.com/newren/git-filter-repo) (`git filter-repo --path allData.zip --invert-paths`) or BFG Repo-Cleaner, then force-push. This rewrites every commit hash, breaks any existing clones/forks, and invalidates commit-SHA references in `HISTORY.md` (e.g. `7c23af2`, `ade9434`) — those would need re-mapping or a caveat note. Only do this if clone size actually matters to you, and do it *before* the repo gets stars/forks you care about.

Either way: never commit large data again — the `.gitignore` already covers `output/`, `dashboard_data/`, and `state/*.pkl`; keep it that way.

### 3.5 `.DS_Store` (2 minutes)
`.DS_Store` is already in `.gitignore` ✅. Just verify none are tracked: `git ls-files | grep DS_Store` — if any appear, `git rm --cached <file>` and commit.

### 3.6 CI badges in README (15 minutes)
CI (pytest + ruff) and docs deploy already run via GitHub Actions, but the README badges are static shields. Add live status badges near the top of `README.md`:

```markdown
![CI](https://github.com/ShaneHurley/BasketballElo/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/ShaneHurley/BasketballElo/actions/workflows/docs.yml/badge.svg)
```

A green "passing" badge is one of the strongest trust signals a recruiter's screen shows.

---

## 4. Contribution-graph habits

Recruiters do look at the green squares, fairly or not. You don't need daily commits — you need *honest, steady* activity:

- **Commit as you go, not in monthly dumps.** The existing history already reads well (real commit messages like *"may have leaks in pre 2026 data"* tell the authentic story). Keep that pattern as roadmap tasks land.
- **Write commit messages like the good ones you already have.** Convention going forward: imperative subject line ≤72 chars, body explaining *why* when non-obvious. E.g. `Add past-only lineup form features behind ablation toggle` — not `update code`. (Your history shows both styles; standardize on the good one.)
- **Use issues and PRs against yourself.** The repo already has roadmap milestone/issue scripts. Working through your own issues and merging via PRs (even solo) demonstrates professional workflow and creates reviewable artifacts recruiters can read.
- **Don't game the graph.** Empty commits or backdated activity are obvious and off-brand for a project whose whole identity is methodological honesty.

---

## 5. Turn the old notebooks into a clean public artifact

The high-school-era notebooks (`NBA ELO marc 2(3).ipynb`, `NBA scrapper.ipynb`, `Player NBA ELO.ipynb`, etc.) are a genuinely compelling origin story — but as a raw folder dump they'd look messy. Options, best first:

1. **A single curated repo: `nba-elo-origins`** — pick 3–4 representative notebooks (first working Elo, the scraper, the stint-update version, the abandoned moving-average branch), add a one-page README framing them as *"the 2021–2024 notebooks that started BasketballElo — preserved as history, not maintained,"* and commit them cleaned (clear outputs or keep small ones). Link it from the profile README's journey section. This converts "old clutter" into "documented five-year growth."
2. **Alternative: GitHub Gists** for individual notebooks with a linking index in the profile README — lighter weight, but no README narrative and no repo to pin.
3. **Don't:** upload the whole `all years all tests` folder as-is (dozens of near-duplicate notebooks, stray CSVs, `im bored` folders). Curate ruthlessly — three great artifacts beat forty confusing ones.

---

## 6. Broader profile polish

- **Profile photo & bio:** professional headshot, bio line like *"Computer Engineering @ Kettering | 6× GM software co-op | Building leak-aware sports analytics in Python"*, location Canton, MI, link to LinkedIn.
- **LinkedIn ↔ GitHub cross-links:** make sure the GitHub URL is on the résumé (it is) and the LinkedIn featured section links the BasketballElo repo *and* the docs site.
- **ATS alignment:** the keywords recruiters/ATS scan for (Python, TypeScript, React, Electron, C/C++, SQL, MATLAB, Docker, ROS 2, CARLA, pytest, GitHub Actions, pandas, Power BI) are all in the profile README's skills section — keep them in sync with the résumé whenever either changes.
- **Stars/follows:** star your own flagship repo (common, harmless), follow people/projects you genuinely use.

---

## 7. Prioritized checklist

### This week (~2 hours total)
- [ ] Create `ShaneHurley/ShaneHurley` repo and paste in `PROFILE_README.md`
- [ ] Set BasketballElo repo description, website link, and topics
- [ ] Pin BasketballElo on your profile
- [ ] Add live CI/docs badges to the BasketballElo README
- [ ] Verify no `.DS_Store` files are tracked (`git ls-files | grep DS_Store`)

### This month
- [ ] Create a 1280×640 social preview image (dashboard screenshot + title) and upload it
- [ ] Tag `v1.0.0` and publish the first GitHub Release with release notes
- [ ] Create the curated `nba-elo-origins` repo (3–4 notebooks + framing README); pin it
- [ ] Decide on `allData.zip`: move to a Release asset (Option A) or schedule a history rewrite (Option B) — and document the decision
- [ ] Update LinkedIn featured section with repo + docs links

### This quarter
- [ ] Work roadmap tasks through self-issues and PRs; tag releases as epics land
- [ ] Write the public post *"Finding chronological leaks in my own NBA model"* (already roadmap task 6.4.1) — publish on the docs site and/or LinkedIn; it's the single best artifact this project can produce for recruiting
- [ ] Re-audit the profile as a stranger: open `github.com/ShaneHurley` in a private window and check the 30-second impression
- [ ] Keep skills badges synchronized with any résumé updates before January 2027 recruiting season

---

*Prepared September 2026 alongside `PROFILE_README.md` and `docs/about.md`. All metrics referenced (2,100+ test definitions, 98 modules, 100+ pytest modules, 50× CoSim acceleration, 38.1 FPS, GPA 3.67, 2,000+ outreach attendees) come from the résumé and this repo's own documentation.*

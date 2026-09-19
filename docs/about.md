# About the Author

**Shane Hurley** — Computer Engineering student at Kettering University (GPA 3.67, expected December 2026), six-time General Motors software engineering co-op, and the author of every line of this project, from the first high-school notebook to the 98-module pipeline you're reading about now.

[:fontawesome-brands-linkedin: LinkedIn](https://linkedin.com/in/shane-hurley-a4b89325a) · [:fontawesome-brands-github: GitHub](https://github.com/ShaneHurley) · [:material-email: shane@hurleyhome.com](mailto:shane@hurleyhome.com)

---

## The journey

### High school: a scraper and a notebook

This project predates my engineering degree. As a high-schooler I taught myself Python by scraping NBA play-by-play data and building a possession-level player Elo system in Google Colab. The first working version — lovingly preserved in notebooks with names like `NBA ELO marc 2(3).ipynb` — tracked separate offensive and defensive Elo ratings for every *player*, keyed off which five-man lineup was on the floor, updating possession-by-possession from raw play-by-play. Flat K-factor, starting rating of 1000, no home-court term, no season regression. Textbook Elo, applied at a granularity most textbook treatments never bother with.

I kept at it through high school and into college: tuning K-factors across near-duplicate notebooks, adding a FIDE-style provisional-rating boost for players with few possessions, moving from single-possession updates to lineup-stint updates, and — in one memorable 2024 branch — abandoning logistic Elo entirely for an exponentially-weighted moving average of points-per-possession. That branch went nowhere, and I went back to Elo. It was a valuable negative result: I learned to let experiments fail on evidence.

### College and six GM rotations

At Kettering University I studied Computer Engineering — embedded systems, operating systems, real-time systems, control systems, computer vision, autonomous driving — while alternating academic terms with six software engineering co-op rotations at General Motors (2022–2026):

- **Connected Vehicle Research / Senior Thesis (2025–2026):** built a reusable CARLA + cockpit-HMI research workflow with optional ROS 2 interfaces; developed simulator regression suites spanning **2,100+ Python test definitions**; replaced routine JSON editing with a 9-tab Electron/browser editor plus MCP/Copilot tooling; integrated React/ProtoPie cockpit displays with a headless benchmark measuring **38.1 FPS**; separately prototyped a TypeScript/Electron human-body-model geometry tool with LS-DYNA node-file I/O.
- **Virtualization Platform & Automation (2024):** automated CoSim setup and Docker/VM script execution, accelerating selected platform tests by up to **50×**.
- **Speech Performance, Tuning & Certification (2024):** made thousands of audio recordings queryable through Python/SQL, with MATLAB validation tools and a cached React interface.
- **Over-the-Air Update Validation (2023):** flashed and troubleshot vehicle ECUs; automated test-rate collection and repeatable setup with Python/Electron tools.
- **Analytics & Business Optimization (2023):** built Power BI dashboards and automated SharePoint/InfoPath workflows for internal clients.
- **Software Defined Vehicle – Web Development (2022):** automated protocol-buffer API-documentation processing with a Python Markdown parser/rewriter.

Those rotations changed how I write everything, including this project. Validation discipline, test-first instincts, and a healthy distrust of any number that looks too good — that's GM's fingerprint on BasketballElo.

### The pipeline you're reading about

In May 2026 I rebuilt the notebook-era model around points-per-possession, dynamic K-factors, and a meta-model layer. A same-season backtest printed **~60% against the spread** — and instead of celebrating, I wrote in the commit message that it *"may be over fit."* That instinct kicked off the most important phase of the project: a deliberate leak hunt. I rewrote the system around Glicko-2-style uncertainty, walk-forward evaluation, and chronological cross-validation; the honest edge claim shrank to a credible 2–4%. Then in July 2026 I decomposed the monolithic notebook into the modular `code/pipeline/` package — **98 modules, 100+ pytest modules**, a formal [leak registry](leaks.md) where every confirmed leak has a root cause, a regression test, and a fix status, a staged full-suite runner, and a local FastAPI T-60 dashboard.

The full timeline lives in [History](history.md).

---

## What this project demonstrates, engineering-wise

- **Leak rigor as a first-class practice.** `PastOnlyGroupCV`, a hand-built `ManualOOFStacker` that raises hard `LeakageError`s on any chronological violation, a SHA-256 fingerprinted `CalibrationSliceRegistry` that provably prevents calibrators from sharing row slices, and point-in-time versioning for external data (EPM priors, injury reports) so a future snapshot can never join to a past game.
- **Honest evaluation.** Walk-forward seasons everywhere; decision-line ≠ closing-line CLV gates; promotion blocked outright without odds provenance. The system is designed to say "I don't know" rather than overclaim.
- **Testing culture.** 100+ pytest modules covering integrity, leaks, evaluation, and the dashboard — including tests that document *old* leaks so they can never silently return. The same habit I practiced at GM scale (2,100+ test definitions) applied to a personal project.
- **Production packaging.** A staged `run_full_suite.py` with checkpoints and anti-overfit review stops, a FastAPI dashboard with SSE progress and Plotly charts, CI via GitHub Actions, and this MkDocs site.
- **Statistical depth.** Multi-method de-vigging (multiplicative / power / odds-ratio / Shin), Venn-Abers bounded calibration, Huber residual calibration chains, James-Stein-style shrinkage, fractional Kelly staking with slate-covariance shrinkage and bankroll Monte Carlo.

---

## Where it's going

The [roadmap](roadmap.md) is a living, research-cited plan — player and lineup rolling form features, EPM/DARKO-style impact-weighted team strength, Kalman-filter now-casting, portfolio-level CVaR staking, and CI/performance hardening — each epic broken into tasks with acceptance criteria, plus an explicit *anti-roadmap* of things I refuse to do (no promoting on ATS/ROI without CLV, no tuning on ROI objectives, no shipping features without past-only isolation tests).

---

## Résumé facts

| | |
|---|---|
| **Education** | B.S. Computer Engineering, Kettering University — expected Dec 2026 |
| **GPA** | 3.67 — Provost Scholarship, Dean's List (six terms) |
| **Experience** | Six General Motors software engineering co-op rotations, Jul 2022 – Sep 2026 |
| **Availability** | Full-time roles starting **January 2027**; U.S. citizen, no sponsorship required; open to relocation |
| **Languages** | Python, TypeScript, JavaScript, C, C++, SQL, MATLAB |
| **Frameworks & tools** | React, Electron, Docker, ROS 2, CARLA 0.9.15, pytest, pandas, GitHub Actions, Power BI, Git, Linux |
| **Leadership** | FIRST Robotics Team 862 mentor (three seasons); led "Robots in the Park" STEM outreach for 2,000+ attendees |
| **Contact** | [linkedin.com/in/shane-hurley-a4b89325a](https://linkedin.com/in/shane-hurley-a4b89325a) · [github.com/ShaneHurley](https://github.com/ShaneHurley) · [shane@hurleyhome.com](mailto:shane@hurleyhome.com) |

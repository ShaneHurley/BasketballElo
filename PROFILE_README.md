# Hi, I'm Shane Hurley 👋

**Computer Engineering senior at Kettering University** (GPA 3.67, expected Dec 2026) · **Six General Motors software engineering co-op rotations** (2022–2026) · **Available for full-time roles starting January 2027** · U.S. citizen, open to relocation

I build software that has to be *right*, not just working — validation tools, data pipelines, and full-stack applications — and I care as much about how a result was produced as the result itself.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shane--hurley-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/shane-hurley-a4b89325a)
[![GitHub](https://img.shields.io/badge/GitHub-ShaneHurley-181717?logo=github&logoColor=white)](https://github.com/ShaneHurley)
[![Email](https://img.shields.io/badge/Email-shane%40hurleyhome.com-EA4335?logo=gmail&logoColor=white)](mailto:shane@hurleyhome.com)

---

## 🏀 Featured project: BasketballElo

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/ShaneHurley/BasketballElo)
[![Tests](https://img.shields.io/badge/tests-100%2B%20pytest%20modules-brightgreen)](https://github.com/ShaneHurley/BasketballElo)
[![Pipeline](https://img.shields.io/badge/pipeline-98%20modules-orange)](https://github.com/ShaneHurley/BasketballElo)
[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-blueviolet)](https://shanehurley.github.io/BasketballElo/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/ShaneHurley/BasketballElo/blob/main/LICENSE)

**[BasketballElo](https://github.com/ShaneHurley/BasketballElo)** — a leak-aware NBA spread-prediction pipeline, five years in the making. What started as a possession-level player Elo notebook is now a modular Python package with Glicko-2 player ratings, walk-forward evaluation, multi-layer calibration, a formal leak registry, and a FastAPI analysis dashboard.

The part I'm proudest of: a backtest once printed ~60% against the spread — and I didn't trust it. I treated the exciting number as a bug report, hunted down a catalog of chronological leaks, built regression tests for each one, and rebuilt until the edge claim was smaller but something I would actually stake my name on. That story — **distrust → catalog → fix → test** — is documented in the repo's [leak registry](https://github.com/ShaneHurley/BasketballElo/blob/main/code/LEAK_REGISTRY.md) and [project history](https://github.com/ShaneHurley/BasketballElo/blob/main/HISTORY.md).

📖 **Full documentation:** [shanehurley.github.io/BasketballElo](https://shanehurley.github.io/BasketballElo/)

---

## 🛣️ My coding journey

I wrote my first real code for this problem before college: as a high-schooler I was scraping NBA play-by-play (`nba_scraper`, one very lovingly named `NBA scrapper.ipynb`) and hand-rolling a possession-level player Elo system in Google Colab — separate offensive and defensive ratings per player, updated lineup-by-lineup from raw play-by-play, with a flat K-factor and a lot of experimentation. Through high school and into Kettering I kept iterating: K-factor tuning, a FIDE-style provisional-rating boost for new players, stint-based updates, and one abandoned detour into moving-average ratings that taught me as much as the successes did.

Six GM co-op rotations later — building CARLA/ROS 2 simulation workflows, Electron/React tooling, and Python/TypeScript validation suites — I came back to the project with professional engineering habits. The notebook became a 98-module tested package with CI, a docs site, and a testing culture where every confirmed data leak gets a named entry and a regression test. The journey from *"cool notebook that scrapes the NBA"* to *"auditable pipeline I can defend line-by-line"* is the through-line of how I've grown as an engineer.

---

## 🧰 Skills

**Languages & frameworks**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![C](https://img.shields.io/badge/C-A8B9CC?logo=c&logoColor=black)
![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?logo=postgresql&logoColor=white)
![MATLAB](https://img.shields.io/badge/MATLAB-FF7F0E?logo=mathworks&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![Electron](https://img.shields.io/badge/Electron-47848F?logo=electron&logoColor=white)

**Tools & platforms**

![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![ROS 2](https://img.shields.io/badge/ROS%202-22314E?logo=ros&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?logo=powerbi&logoColor=black)

**Domains:** simulation & validation (CARLA 0.9.15, CoSim, HMI, telemetry/replay), OTA/ECU validation, automated testing, data pipelines, full-stack internal tools

---

## 💼 Experience highlights (General Motors, six rotations, 2022–2026)

- **Connected Vehicle Research / Senior Thesis** — reusable CARLA + cockpit-HMI research workflow with optional ROS 2 interfaces; simulator regression suites spanning **2,100+ Python test definitions**; a 9-tab Electron/browser editor with MCP/Copilot tooling; headless rendering benchmark of **38.1 FPS**
- **Virtualization Platform & Automation** — automated CoSim setup and Docker/VM execution, accelerating selected platform tests by up to **50×**
- **Speech Performance & Certification** — made thousands of audio recordings queryable via Python/SQL with MATLAB validation and a cached React interface
- **OTA Update Validation** — flashed and troubleshot vehicle ECUs; automated test-rate collection with Python/Electron tools
- **Analytics & Business Optimization** — Power BI dashboards and automated SharePoint/InfoPath workflows for internal clients
- **Software Defined Vehicle** — automated protocol-buffer API-documentation processing with a Python Markdown parser/rewriter

---

## 🤖 Leadership & service

- **FIRST Robotics Team 862** — mentored robot design, build, and programming for **three seasons**
- **Robots in the Park** — led a STEM outreach event serving **2,000+ attendees**

---

## 🗺️ What's next for BasketballElo

The project has a living, research-backed roadmap — player/lineup rolling form features, EPM/DARKO-style impact-weighted team strength, Kalman-filter now-casting, portfolio-level staking, and CI/performance hardening — each task with acceptance criteria and explicit anti-goals.

- 📋 [Roadmap (docs/roadmap.md)](https://github.com/ShaneHurley/BasketballElo/blob/main/docs/roadmap.md)
- 🌐 [Documentation site](https://shanehurley.github.io/BasketballElo/)
- 📜 [Five-year project history](https://github.com/ShaneHurley/BasketballElo/blob/main/HISTORY.md)

---

## 📫 Get in touch

I'm graduating in **December 2026** and available for full-time software engineering roles starting **January 2027**. If you work on validation tooling, data pipelines, simulation, or anything where correctness matters, I'd love to talk.

- 💼 [linkedin.com/in/shane-hurley-a4b89325a](https://linkedin.com/in/shane-hurley-a4b89325a)
- 🐙 [github.com/ShaneHurley](https://github.com/ShaneHurley)
- ✉️ [shane@hurleyhome.com](mailto:shane@hurleyhome.com)

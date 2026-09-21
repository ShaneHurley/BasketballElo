# This copy is not production

The canonical pipeline package is the repo-root `pipeline/` directory
(`/Users/shurley/Documents/basketball/pipeline`).

Root `conftest.py` and `tests/test_production_tree.py` force every
`import pipeline` used by production scripts and the root test suite to
resolve there. Do not add new leak fixes only in this tree.

# FaceKit Documentation

Reference documentation for FaceKit. The top-level `../README.md` covers
installation and the CLI overview; the documents here go deeper.

| Document | What it covers |
| --- | --- |
| [`WORKED_EXAMPLE.md`](WORKED_EXAMPLE.md) | A full end-to-end walkthrough — install, `extract-landmarks`, `extract-features` — using real commands and real output. Start here if you are new. |
| [`OUTPUT_FORMAT.md`](OUTPUT_FORMAT.md) | Exact file types and column/field schemas produced by `extract-landmarks` (JSON / JSONL) and `extract-features` (phenotype CSV). |
| [`FEATURE_GLOSSARY.md`](FEATURE_GLOSSARY.md) | What each of the 120 geometric feature columns means — units, sign conventions, and the nine feature families. |
| [`SYNTHETIC.md`](SYNTHETIC.md) | The generation pipeline — `enhance`, `pack`, `train`, `generate` — every default and its origin, and the six changes to the vendored StyleGAN3. |
| [`PRIVACY.md`](PRIVACY.md) | The privacy analysis — inputs, the two axes (identity, appearance), the flagging and NNAA statistics, output files, and how to read the manuscript's result. |

Suggested reading order for a new user: **Worked Example → Output Format →
Feature Glossary.**

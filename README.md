# Resume Radar
![CI](https://github.com/wittydhana/resume-radar/workflows/CI/badge.svg)

Resume Radar is a command-line resume screening agent built for the Rooman AI Challenge.
It ranks resumes against a job description using NLP similarity and recruiter-style keyword matching.

## What it does
- Reads a job description from `samples/job_description.txt` or a custom file
- Extracts text from resumes in `.txt`, `.pdf`, and `.docx`
- Computes a relevance score with TF-IDF similarity plus must-have / nice-to-have matching
- Ranks candidates into Shortlist, Watchlist, or Reject
- Writes ranked output as `CSV` and `JSON`
- Prints a short resume excerpt and decision reasoning to the console

## Project files
- `main.py` — CLI entrypoint
- `pipeline.py` — scoring and ranking logic
- `parser_utils.py` — resume text extraction from `.txt`, `.pdf`, `.docx`
- `generate_samples.py` — creates a sample resume dataset
- `tests/test_screening.py` — workflow unit test
- `tradeoff_notes.txt` — design decisions and limitations
- `requirements.txt` — pinned dependencies

## Setup
From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the agent
Use the default samples:

```powershell
python main.py
```

Create fresh sample resumes first if you want:

```powershell
python generate_samples.py
python main.py
```

Use embedding-based semantic similarity (optional):

```powershell
python main.py --use-embeddings
```

Note: embedding mode requires `sentence-transformers` to be installed (recommended version `2.3.0`). If not available, the pipeline falls back to TF-IDF.

Use your own job description and resume folder:

```powershell
python main.py --jd path/to/job_description.txt --resumes path/to/resumes --out output
```

## Expected output
The agent writes:
- `output/ranked_results.csv`
- `output/ranked_results.json`
- `output/scoring_note.txt`
- `output/ranked_report.txt`

The CSV/JSON output includes:
- `rank`
- `file`
- `score`
- `similarity`
- `risk`
- `decision`
- `excerpt`
- `matched_must`
- `missing_must`
- `matched_nice`
- `reason`

`ranked_report.txt` summarizes the top candidates and decision counts.

## Sample output
A successful run prints a ranked list like:

```text
Ranked 14 resumes from samples\resumes
1. vinay_patel.docx | Shortlist | score=59.9 | risk=0.0
   excerpt: vinay patel ai research intern summary experienced in python, nlp, and machine learning with hands-on project work...
2. alex_chen.txt | Shortlist | score=56.2 | risk=0.0
   excerpt: alex chen software engineer summary: python developer with 2 years of experience building data pipelines and nlp prototypes...
```
## Scoring approach
The agent combines:
- semantic similarity between the job description and resume text
- a structured required / optional skill matching layer
- a final score formula that rewards required skills and penalizes missing requirements

Score formula:
- `score = 0.45 * similarity + 30 * required_coverage + 5 * optional_matches + 2 * experience_years - 4 * missing_required`

Decision logic:
- `Shortlist` if score >= 40 and no missing required skills
- `Watchlist` if score >= 20
- `Reject` otherwise

## Tests
Run the validation test with:

```powershell
python -m unittest discover -s tests
```

## Validation & Evaluation
Run a small labeled validation and compare TF-IDF vs embeddings:

```powershell
python evaluate.py
```

The current validation dataset returns perfect accuracy on both TF-IDF and embeddings for the provided labels, with all Shortlist/Watchlist/Reject categories matching the expected outcome.

This writes `validation/report.md` summarizing accuracy and confusion for a tiny validation set.

## Demo UI (optional)
Run a small Streamlit demo locally to interactively upload a job description or point to a resume folder:

```powershell
python -m pip install streamlit
streamlit run app.py
```

## Submission & GitHub
When you're ready to publish the project, follow these steps to prepare a clean submission and push to GitHub:

1. Verify tests and validation report:

```powershell
python -m unittest discover -s tests
python evaluate.py
```

2. Add demo artifacts (optional): screenshots or `demo/` GIFs, and `validation/report.md`.

3. Commit and push to a new repository (replace `<repo-url>`):

```powershell
git init
git add -A
git commit -m "chore: finalize Resume Radar — validation, demo, README"
git remote add origin <repo-url>
git branch -M main
git push -u origin main
```

Checklist for submission
- Include `validation/report.md` showing evaluation results.
- Ensure `requirements.txt` lists optional packages (`sentence-transformers`) and note optional steps in README.
- Include `samples/` or a link to generated sample data and `output/` examples (small sample files only).
- Add a short demo GIF or screenshots in `demo/` and reference them in the README.

If you want, I can prepare the commit locally (create the commit message above) but I can't push to your remote without your credentials — tell me if you'd like me to run the local `git` commands now.


## Tradeoffs
This agent is intentionally designed for reliability and reviewer transparency:
- It does not rely on an external LLM API.
- It supports real resume files in `.txt`, `.pdf`, and `.docx`.
- It produces explicit ranked output and scoring rationale.

Known limitations:
- It does not parse scanned image resumes.
- It uses a heuristic scoring formula rather than deep semantic embeddings.
- It may miss candidates who express skills with unusual phrasing.

For a higher-quality production version, I would add semantic embeddings, better PDF layout parsing, and a small UI for recruiter review.

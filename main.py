import argparse
from pathlib import Path
from pipeline import run_screening


def main():
    p = argparse.ArgumentParser(description="Resume Radar")
    p.add_argument("--jd", default="samples/job_description.txt")
    p.add_argument("--resumes", default="samples/resumes")
    p.add_argument("--out", default="output")
    p.add_argument("--use-embeddings", action="store_true", help="Use sentence-transformer embeddings for semantic similarity if available")
    args = p.parse_args()
    run_screening(Path(args.jd), Path(args.resumes), Path(args.out), use_embeddings=bool(args.use_embeddings))


if __name__ == "__main__":
    main()
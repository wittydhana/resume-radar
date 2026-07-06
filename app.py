import streamlit as st
from pathlib import Path
from pipeline import run_screening

st.title('Resume Radar — Demo')

jd = st.text_area('Job description (paste text or leave blank to use samples/job_description.txt)')
resumes_dir = st.text_input('Resumes folder (path)', value='samples/resumes')
use_emb = st.checkbox('Use embeddings (if installed)', value=False)

if st.button('Run screening'):
    if not jd.strip():
        jd_path = Path('samples/job_description.txt')
    else:
        tmp = Path('tmp_jd.txt')
        tmp.write_text(jd, encoding='utf-8')
        jd_path = tmp
    out = Path('output')
    df = run_screening(jd_path, Path(resumes_dir), out, use_embeddings=use_emb)
    st.write(df)
    st.markdown('### Top candidates')
    for _, r in df.head(5).iterrows():
        st.write(f"{r['rank']}. {r['file']} — {r['decision']} — score={r['score']}")
        st.write(r['excerpt'][:300])

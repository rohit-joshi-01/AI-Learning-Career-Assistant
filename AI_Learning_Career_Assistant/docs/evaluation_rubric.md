# Evaluation Rubric — Total: 100 Marks

## Scoring Breakdown

| Category | Criteria | Marks |
|---|---|---|
| **Recommendation Engine (30)** | Working vector search + retrieval | 10 |
| | Scoring / ranking logic implemented | 10 |
| | Confidence score with documented formula | 10 |
| **Embeddings & Vector DB (20)** | Embeddings correctly generated | 10 |
| | ChromaDB (or similar) properly integrated | 10 |
| **LLM Integration (15)** | Gemini API used for explanation only (not selection) | 8 |
| | Roadmap and interview tips generated correctly | 7 |
| **Skill Gap Analysis (10)** | Gap identified correctly vs. top recommendations | 10 |
| **UI / UX (10)** | Streamlit app is usable and shows all outputs clearly | 10 |
| **Documentation (10)** | README complete, architecture documented, clear setup | 10 |
| **Security & Responsible AI (5)** | No exposed keys, PII handled, disclaimer present | 5 |
| **Bonus: Deployment (+5)** | Live working deployment URL submitted | +5 |

## Penalty Table

| Violation | Penalty |
|---|---|
| API key found in code or commit history | −20 marks + security flag |
| Hardcoded recommendations detected | −15 marks |
| LLM used for recommendation selection | −15 marks |
| Copied GitHub code (plagiarism confirmed) | 0 marks, disqualification |
| No confidence score in output | −10 marks |
| Fabricated / screenshot-only demo | 0 marks for demo section |
| Missing README or incomplete setup | −5 marks |
| Missing dataset files | −5 marks |
| No `.env.example` file | −3 marks |

## Mentor Verification Checklist

### Core Technical Checks
- [ ] Vector search returns results from ChromaDB (not hardcoded)
- [ ] Confidence score is computed from the 4 components
- [ ] LLM is NOT deciding which jobs to recommend
- [ ] Skill gap analysis shows missing skills per recommendation
- [ ] Resume upload and text extraction works on a real PDF

### Security Checks
- [ ] No API keys in any code file
- [ ] `git log --all -p | grep -i api_key` returns nothing
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` is present

### Documentation Checks
- [ ] README has all 8 required sections
- [ ] Architecture diagram is present
- [ ] Confidence formula is documented with weights
- [ ] Dataset sources and licenses are listed

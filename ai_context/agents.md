# Strawberry Codex Agent Context (AGENTS.md)

## Purpose
This file is the **entry point** for all AI context in `ai_context`. Use it to find the authoritative document for any task and to resolve conflicts between documents.

**Financial-modelling** is a  Streamlit-based educational app for learning core financial modeling concepts. It uses interactive explanations, visualizations, and simple Python models to teach topics like cash flows, valuation, risk and return, and scenario analysis, with a focus on clarity and hands-on learning.

---

## 1) Start here (primary sources)
Use these documents first, in order, when implementing or updating code:

1. **Repo/runtime/tooling defaults**
   - `docs/AI_context/technology_stack.md`

2. **UI ops console (Streamlit)**
   - `docs/AI_context/ui_context.md`

---

## 2) Hard constraints (do not violate)
1. 

---

## 3) Conflict resolution
If two documents appear to conflict, prefer (in order):
1. 

---

## 5) Working rules for generated changes
### 5.1 Style and typing
- Python >= 3.12
- Use built-in generics (`list[...]`, `dict[...]`) and strict-ish typing
- Keep functions deterministic and testable; avoid hidden side effects

---

## 6) What “done” looks like
For code changes produced by Codex:
- Changes compile/type-check at a reasonable baseline (mypy-friendly)
- Formatting consistent with repo standards (Black/isort/flake8)
- No contract violations 
- Where meaningful, include a small test fixture or example usage


## 7) ExecPlans
 
When writing complex features or significant refactors, use an ExecPlan (as described in `ai_context/plans.md`) from design to implementation.

END OF DOCUMENT

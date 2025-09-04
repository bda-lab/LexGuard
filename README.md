# LexGuard
# Policy Compliance Verification System (Offline RAG-LLM Web App)

## Problem Statement

We aim to develop an **offline, LLM-powered web application** for automated policy compliance verification across heterogeneous contexts, such as **GDPR-related regulations between countries, contractual clauses in internship agreements versus institutional policy documents, or HR guidelines (e.g., leave policies) against organizational rulebooks**.  

Since the documents involved are often **large, sensitive, and security-critical**, they cannot be shared with external online LLM services. Moreover, their size may **exceed the native context window** of modern language models, necessitating the integration of a **retrieval-augmented generation (RAG) pipeline**.  

In this setup, policy documents would be ingested into a **vector database** (or equivalent retrieval layer), enabling efficient semantic search to dynamically retrieve only the most relevant segments for context construction during queries. A crucial challenge lies in **domain-aware vectorization**, where embeddings must be generated with respect to the compliance-checking objectives rather than generic semantic similarity.  

The system should be designed as a **modular, API-driven architecture**, where components (e.g., embedding service, retrieval engine, reasoning agent, compliance evaluator) remain **loosely coupled** to allow easy substitution of LLMs or AI agents without disrupting the overall workflow.

---

## Team Structure & Responsibilities

### **Student A — Data Collection, Curation & Governance**
- Acquire GDPR texts, institutional policies, HR manuals, contracts, etc.
- Redaction, de-duplication, versioning, and schema design.
- Build a labeled dataset for evaluation.
- Deliverables: `datasets/`, schema/ontology, annotation guidelines, data card.

### **Student B — Ingestion, Chunking, Embedding & Retrieval**
- Implement document parsers (PDF/DOCX/HTML).
- Domain-aware chunking + embeddings.
- Setup vector database + retrieval pipeline (semantic + keyword hybrid).
- Deliverables: Ingestion service, vector DB, retrieval evaluation report.

### **Student C — Reasoning, Compliance Engine & Evaluation**
- Design decision schema (status, evidence, rationale, confidence).
- Develop compliance assessment engine (prompting + rule library).
- Build evaluation harness with precision/recall, evidence alignment metrics.
- Deliverables: Compliance engine API, evaluation reports, error analysis.

### **Student D — Offline Web App, APIs & Deployment**
- Build offline web UI (upload, search, compare, assess).
- Develop API gateway for modular services.
- Package everything in Docker Compose for offline deployment.
- Deliverables: Web UI, REST APIs, deployment scripts, observability dashboards.

---
## Notion Page
https://rust-mandolin-74e.notion.site/LexGuard-25ca338f8c5b80aca495c55c3bdc8ea2?pvs=74

---
## Project Milestones

### **Week 1–3 — Foundations**
Identified references: 
[28/08/25, 11:19:15 AM] Vinu: https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf
[28/08/25, 11:19:30 AM] Vinu: https://aclanthology.org/2025.coling-main.178.pdf

Discuss the above paper and expand on the ideas—similar to the negation documents we discussed, what other strategies could be explored to improve the system’s accuracy? Identify and discuss relevant datasets in this context, and provide a concrete example based on the Digital Personal Data Protection Act, 2023.

### **Week 4–6 — RAG & Engine v1**


### **Week 7–8 — **


### **Week 9–11 — **


---
### Participation

- Subham  ||
- Trupti  |
- Divyam  | 
- Owais   |

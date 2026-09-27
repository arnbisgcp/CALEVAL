# Healthcare Claims Agent (`google-adk` on Vertex AI Agent Engine)

An end-to-end **Google Agent Development Kit (ADK)** agent (`gemini-2.5-flash`) deployed on **Vertex AI Agent Engine** (`reasoningEngines`) in Google Cloud project **`arnbtest`**, complete with realistic healthcare claims mock data, 6 adjudication & member service tools, and a standalone interactive web chat console.

---

## 1. Deployed GCP Resource Details

- **Project ID:** `arnbtest` (`projects/163474459793`)
- **Region:** `us-central1`
- **Reasoning Engine Resource:** `projects/163474459793/locations/us-central1/reasoningEngines/6960836041880109056`
- **Agent Registry Resource:** `projects/arnbtest/locations/us-central1/agents/agentregistry-00000000-0000-0000-6002-f664f5cff1ee`
- **Model:** `gemini-2.5-flash`
- **Framework:** `google-adk` (`2.6.3`)
- **GCP Console Playground URL:**
  `https://console.cloud.google.com/vertex-ai/agents/agent-engines/locations/us-central1/agent-engines/6960836041880109056?project=arnbtest`

---

## 2. Repository Structure

```text
.
├── main.py                # ADK Healthcare Claims Agent definition, 6 tools, and mock dataset
├── requirements.txt       # Vertex AI Agent Engine runtime dependencies
├── deploy_agent.py        # Source-based deployment script for Vertex AI Reasoning Engine
├── test_agent.py          # Live SSE streaming verification client (:streamQuery)
└── web_ui/                # Interactive Adjudication & Member Resolution Web Chat Console
    ├── server.py          # Local HTTP server & Vertex AI Reasoning Engine proxy
    ├── index.html         # Web console layout & interactive mock directory
    ├── styles.css         # Clinical styling & responsive layout
    └── app.js             # Chat logic, live ADK tool trace inspector, and session state
```

---

## 3. ADK Tools & Mock Dataset

### ADK Tools (`main.py`)
1. `get_member_eligibility(member_id_or_name)` — Look up member coverage status, plan details, deductible/OOP max accumulators, coinsurance, and copays.
2. `list_member_claims(member_id, status_filter="ALL")` — List claims for a member filtered by status (`ALL`, `PAID`, `DENIED`, `PENDING_REVIEW`).
3. `get_claim_details(claim_id)` — Inspect full CPT/ICD-10 line-item adjudication, CARC/RARC denial codes, EOB numbers, and appeal deadlines.
4. `check_prior_authorization(member_id="", pa_id="")` — Check prior authorization status, approved CPT codes, and approved provider NPIs.
5. `estimate_patient_responsibility(member_id, cpt_code, estimated_allowed_amount, in_network=True)` — Calculate estimated patient out-of-pocket cost based on remaining deductible, coinsurance, and out-of-pocket maximum.
6. `submit_claim_appeal(claim_id, appeal_reason, supporting_reference="")` — Submit a formal claim appeal for a denied claim and generate a tracking ID.

### Mock Members & Claims
| Member ID | Name | Plan | Status | Claims | Prior Authorizations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MEM-1001` | **Sarah Jenkins** | Gold PPO Plus | `ACTIVE` | `CLM-2026-9001` (`PAID`)<br>`CLM-2026-9002` (`DENIED` `CO-197`) | `PA-2026-441` (`APPROVED` for CPT `29881`) |
| `MEM-1002` | **Michael Chen** | Silver HMO Select | `ACTIVE` | `CLM-2026-9003` (`PAID`)<br>`CLM-2026-9004` (`PENDING_REVIEW`) | `PA-2026-512` (`PENDING_CLINICAL_REVIEW` for CPT `71250`) |
| `MEM-1003` | **Elena Rodriguez** | Platinum EPO Premier | `ACTIVE` | `CLM-2026-9005` (`DENIED` `CO-50`) | — |
| `MEM-1004` | **David Ross** | Bronze HDHP HSA | `INACTIVE` | — | — |

---

## 4. Running & Testing

### Test the Deployed Agent via CLI
```bash
python3 test_agent.py
```

### Launch the Interactive Web Chat Console
```bash
python3 web_ui/server.py
```
Then open `http://localhost:8085` (or `http://arnabtest.c.googlers.com:8085` on Cloudtop).

### Re-deploy to Vertex AI Agent Engine
```bash
tar -czf source.tar.gz main.py requirements.txt
python3 deploy_agent.py
```

"""Persistent SQLite Relational Database & Clinical Policy Vector Store for Healthcare Claims Agent."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
import sqlite3
import threading
from typing import Any, Optional

from schemas import (
    AppealRecord,
    ClaimRecord,
    MemberRecord,
    PolicyDocumentMatch,
    PriorAuthorizationRecord,
)

DEFAULT_DB_PATH = os.environ.get(
    "HEALTHCARE_CLAIMS_DB_PATH", "/tmp/healthcare_claims_store.db"
)

# ============================================================================
# Canonical Seed Data (Also exported for UI & Golden Evaluation Reference)
# ============================================================================

MOCK_MEMBERS: dict[str, dict[str, Any]] = {
    "MEM-1001": {
        "member_id": "MEM-1001",
        "full_name": "Sarah Jenkins",
        "dob": "1985-04-12",
        "plan_name": "Gold PPO Plus",
        "group_number": "GRP-88420",
        "status": "ACTIVE",
        "effective_date": "2026-01-01",
        "deductible_individual": 1500.00,
        "deductible_met": 1200.00,
        "oop_max_individual": 5000.00,
        "oop_max_met": 2150.00,
        "coinsurance_in_network": 0.20,
        "coinsurance_out_of_network": 0.40,
        "copays": {
            "pcp": 20.00,
            "specialist": 40.00,
            "urgent_care": 50.00,
            "emergency_room": 250.00,
        },
    },
    "MEM-1002": {
        "member_id": "MEM-1002",
        "full_name": "Michael Chen",
        "dob": "1978-11-03",
        "plan_name": "Silver HMO Select",
        "group_number": "GRP-55190",
        "status": "ACTIVE",
        "effective_date": "2026-01-01",
        "deductible_individual": 3000.00,
        "deductible_met": 650.00,
        "oop_max_individual": 7500.00,
        "oop_max_met": 950.00,
        "coinsurance_in_network": 0.30,
        "coinsurance_out_of_network": 0.50,
        "copays": {
            "pcp": 30.00,
            "specialist": 60.00,
            "urgent_care": 75.00,
            "emergency_room": 350.00,
        },
    },
    "MEM-1003": {
        "member_id": "MEM-1003",
        "full_name": "Elena Rodriguez",
        "dob": "1992-07-24",
        "plan_name": "Platinum EPO Premier",
        "group_number": "GRP-90311",
        "status": "ACTIVE",
        "effective_date": "2026-01-01",
        "deductible_individual": 500.00,
        "deductible_met": 500.00,
        "oop_max_individual": 3000.00,
        "oop_max_met": 1480.00,
        "coinsurance_in_network": 0.10,
        "coinsurance_out_of_network": 1.00,
        "copays": {
            "pcp": 15.00,
            "specialist": 30.00,
            "urgent_care": 40.00,
            "emergency_room": 150.00,
        },
    },
    "MEM-1004": {
        "member_id": "MEM-1004",
        "full_name": "David Ross",
        "dob": "1964-02-19",
        "plan_name": "Bronze HDHP HSA",
        "group_number": "GRP-11204",
        "status": "INACTIVE",
        "effective_date": "2025-01-01",
        "termination_date": "2025-12-31",
        "cobra_eligible": True,
        "deductible_individual": 6000.00,
        "deductible_met": 0.00,
        "oop_max_individual": 8000.00,
        "oop_max_met": 0.00,
        "coinsurance_in_network": 0.20,
        "coinsurance_out_of_network": 0.50,
        "copays": {
            "pcp": 0.00,
            "specialist": 0.00,
            "urgent_care": 0.00,
            "emergency_room": 0.00,
        },
    },
}

MOCK_CLAIMS: dict[str, dict[str, Any]] = {
    "CLM-2026-9001": {
        "claim_id": "CLM-2026-9001",
        "member_id": "MEM-1001",
        "patient_name": "Sarah Jenkins",
        "service_date": "2026-08-14",
        "received_date": "2026-08-16",
        "processed_date": "2026-08-21",
        "provider_name": "Pacific Orthopedic Associates",
        "provider_npi": "NPI-1234567890",
        "network_status": "IN_NETWORK",
        "diagnosis_codes": [
            {
                "icd10": "M25.561",
                "description": "Pain in right knee",
            },
            {
                "icd10": "M23.211",
                "description": (
                    "Derangement of anterior horn of medial meniscus, right knee"
                ),
            },
        ],
        "line_items": [
            {
                "cpt": "99214",
                "description": (
                    "Office or other outpatient visit, established patient (30-39 min)"
                ),
                "billed": 220.00,
                "allowed": 140.00,
                "plan_paid": 100.00,
                "patient_resp": 40.00,
                "notes": "Specialist copay applied ($40.00)",
            },
            {
                "cpt": "73721",
                "description": (
                    "MRI any joint of lower extremity without contrast material"
                ),
                "billed": 1450.00,
                "allowed": 980.00,
                "plan_paid": 764.00,
                "patient_resp": 216.00,
                "notes": "$20.00 applied to remaining deductible + 20% coinsurance ($196.00)",
            },
        ],
        "total_billed": 1670.00,
        "total_allowed": 1120.00,
        "total_plan_paid": 864.00,
        "total_patient_responsibility": 256.00,
        "status": "PAID",
        "eob_number": "EOB-88412",
        "denial_code": None,
        "denial_reason": None,
        "appeal_eligible": False,
    },
    "CLM-2026-9002": {
        "claim_id": "CLM-2026-9002",
        "member_id": "MEM-1001",
        "patient_name": "Sarah Jenkins",
        "service_date": "2026-09-02",
        "received_date": "2026-09-04",
        "processed_date": "2026-09-09",
        "provider_name": "Bay Area Surgical Center",
        "provider_npi": "NPI-1987654321",
        "network_status": "IN_NETWORK",
        "diagnosis_codes": [
            {
                "icd10": "M23.211",
                "description": (
                    "Derangement of anterior horn of medial meniscus, right knee"
                ),
            }
        ],
        "line_items": [
            {
                "cpt": "29881",
                "description": (
                    "Arthroscopy, knee, surgical; with meniscectomy (medial OR lateral)"
                ),
                "billed": 4800.00,
                "allowed": 3200.00,
                "plan_paid": 0.00,
                "patient_resp": 4800.00,
                "notes": "Denied: CO-197 Precertification/authorization absent",
            }
        ],
        "total_billed": 4800.00,
        "total_allowed": 3200.00,
        "total_plan_paid": 0.00,
        "total_patient_responsibility": 4800.00,
        "status": "DENIED",
        "eob_number": "EOB-89205",
        "denial_code": "CO-197",
        "denial_reason": (
            "Precertification/authorization/notification absent. Note: Prior "
            "Authorization PA-2026-441 was approved for CPT 29881 under "
            "Pacific Orthopedic Associates (NPI-1234567890), but claim was "
            "billed under surgical facility NPI-1987654321."
        ),
        "appeal_eligible": True,
        "appeal_deadline": "2027-03-01",
    },
    "CLM-2026-9003": {
        "claim_id": "CLM-2026-9003",
        "member_id": "MEM-1002",
        "patient_name": "Michael Chen",
        "service_date": "2026-09-10",
        "received_date": "2026-09-12",
        "processed_date": "2026-09-18",
        "provider_name": "Metro Pulmonary & Sleep Clinic",
        "provider_npi": "NPI-1445566778",
        "network_status": "IN_NETWORK",
        "diagnosis_codes": [
            {
                "icd10": "J45.40",
                "description": "Moderate persistent asthma, uncomplicated",
            }
        ],
        "line_items": [
            {
                "cpt": "99213",
                "description": (
                    "Office or other outpatient visit, established patient (20-29 min)"
                ),
                "billed": 160.00,
                "allowed": 110.00,
                "plan_paid": 50.00,
                "patient_resp": 60.00,
                "notes": "Specialist copay ($60.00)",
            },
            {
                "cpt": "94010",
                "description": (
                    "Spirometry, including graphic record, total and timed vital capacity"
                ),
                "billed": 120.00,
                "allowed": 85.00,
                "plan_paid": 59.50,
                "patient_resp": 25.50,
                "notes": "30% coinsurance applied",
            },
        ],
        "total_billed": 280.00,
        "total_allowed": 195.00,
        "total_plan_paid": 109.50,
        "total_patient_responsibility": 85.50,
        "status": "PAID",
        "eob_number": "EOB-90114",
        "denial_code": None,
        "denial_reason": None,
        "appeal_eligible": False,
    },
    "CLM-2026-9004": {
        "claim_id": "CLM-2026-9004",
        "member_id": "MEM-1002",
        "patient_name": "Michael Chen",
        "service_date": "2026-09-20",
        "received_date": "2026-09-22",
        "processed_date": None,
        "provider_name": "Golden Gate Diagnostic Imaging",
        "provider_npi": "NPI-1778899001",
        "network_status": "IN_NETWORK",
        "diagnosis_codes": [
            {"icd10": "R91.8", "description": "Other nonspecific abnormal finding of lung field"}
        ],
        "line_items": [
            {
                "cpt": "71250",
                "description": "Computed tomography, thorax, diagnostic; without contrast material",
                "billed": 1100.00,
                "allowed": 680.00,
                "plan_paid": 0.00,
                "patient_resp": 0.00,
                "notes": "Pending medical necessity review & prior auth matching (PA-2026-512)",
            }
        ],
        "total_billed": 1100.00,
        "total_allowed": 680.00,
        "total_plan_paid": 0.00,
        "total_patient_responsibility": 0.00,
        "status": "PENDING_REVIEW",
        "eob_number": None,
        "denial_code": None,
        "denial_reason": (
            "Pended for clinical attachment verification against PA-2026-512. "
            "Expected adjudication completion within 5 business days."
        ),
        "appeal_eligible": False,
    },
    "CLM-2026-9005": {
        "claim_id": "CLM-2026-9005",
        "member_id": "MEM-1003",
        "patient_name": "Elena Rodriguez",
        "service_date": "2026-09-05",
        "received_date": "2026-09-07",
        "processed_date": "2026-09-14",
        "provider_name": "Precision Genomics Lab",
        "provider_npi": "NPI-1554433221",
        "network_status": "OUT_OF_NETWORK",
        "diagnosis_codes": [
            {"icd10": "Z13.79", "description": "Encounter for other screening for genetic and chromosomal anomalies"}
        ],
        "line_items": [
            {
                "cpt": "81479",
                "description": "Unlisted molecular pathology procedure (Whole Exome Panel)",
                "billed": 3400.00,
                "allowed": 0.00,
                "plan_paid": 0.00,
                "patient_resp": 3400.00,
                "notes": "Denied: CO-50 Non-covered service / experimental or not medically necessary for screening ICD-10 Z13.79",
            }
        ],
        "total_billed": 3400.00,
        "total_allowed": 0.00,
        "total_plan_paid": 0.00,
        "total_patient_responsibility": 3400.00,
        "status": "DENIED",
        "eob_number": "EOB-89550",
        "denial_code": "CO-50",
        "denial_reason": (
            "These are non-covered services because this is not deemed a "
            "medical necessity by the payer. CPT 81479 requires a documented "
            "diagnostic ICD-10 code and letter of medical necessity rather "
            "than routine screening code Z13.79, plus Platinum EPO does not "
            "cover out-of-network non-emergent labs."
        ),
        "appeal_eligible": True,
        "appeal_deadline": "2027-03-14",
    },
}

MOCK_PRIOR_AUTHS: dict[str, dict[str, Any]] = {
    "PA-2026-441": {
        "pa_id": "PA-2026-441",
        "member_id": "MEM-1001",
        "patient_name": "Sarah Jenkins",
        "cpt_code": "29881",
        "procedure_description": "Arthroscopy, knee, surgical; with meniscectomy",
        "status": "APPROVED",
        "approved_provider": "Pacific Orthopedic Associates",
        "approved_npi": "NPI-1234567890",
        "valid_from": "2026-08-25",
        "valid_to": "2026-11-25",
        "notes": (
            "Approved for 1 surgical procedure. Facility NPI can be updated to "
            "Bay Area Surgical Center (NPI-1987654321) via claim appeal or "
            "retro-auth request."
        ),
    },
    "PA-2026-512": {
        "pa_id": "PA-2026-512",
        "member_id": "MEM-1002",
        "patient_name": "Michael Chen",
        "cpt_code": "71250",
        "procedure_description": "CT Thorax without contrast",
        "status": "PENDING_CLINICAL_REVIEW",
        "approved_provider": "Golden Gate Diagnostic Imaging",
        "approved_npi": "NPI-1778899001",
        "valid_from": None,
        "valid_to": None,
        "notes": "Awaiting chest X-ray report from ordering pulmonologist.",
    },
}

CLINICAL_POLICY_DOCUMENTS: list[dict[str, Any]] = [
    {
        "policy_id": "POL-CARC-197",
        "title": "CARC CO-197: Precertification, Authorization, and Rendering/Facility NPI Alignment Policy",
        "category": "Administrative & Prior Authorization Adjudication",
        "cpt_codes": ["29881", "29880", "27447", "71250"],
        "carc_codes": ["CO-197"],
        "content": (
            "Under Plan Policy POL-CARC-197, claims denied with CARC code CO-197 "
            "(Precertification/authorization/notification absent) where an active "
            "Prior Authorization exists for the same member and CPT code under the "
            "ordering/surgical group NPI (e.g., Pacific Orthopedic Associates, "
            "NPI-1234567890) but was billed by an in-network Ambulatory Surgical "
            "Center (e.g., Bay Area Surgical Center, NPI-1987654321) qualify for "
            "administrative retro-authorization override. Adjusters or members may "
            "submit a Level-1 Claim Appeal referencing the approved PA number "
            "(e.g., PA-2026-441) to link the facility NPI and reprocess the claim "
            "at the in-network allowed rate within 10 business days."
        ),
    },
    {
        "policy_id": "POL-LCD-L33965",
        "title": "CMS LCD L33965: Surgical Knee Arthroscopy and Meniscectomy (CPT 29881) Medical Necessity",
        "category": "Orthopedic Surgery Clinical Guidelines",
        "cpt_codes": ["29881", "73721"],
        "carc_codes": ["CO-197", "CO-50"],
        "content": (
            "Arthroscopic knee meniscectomy (CPT 29881) is medically necessary when "
            "the member presents with documented mechanical knee symptoms (locking, "
            "catching, or persistent right/left knee pain ICD-10 M25.561) and MRI "
            "(CPT 73721) confirmation of medial or lateral meniscus derangement "
            "(ICD-10 M23.211). Prior authorization is valid for 90 days from issue."
        ),
    },
    {
        "policy_id": "POL-CARC-50-GENOMICS",
        "title": "CARC CO-50: Molecular Pathology & Unlisted Genetic Panels (CPT 81479) Coverage Criteria",
        "category": "Molecular Diagnostics & EPO Network Policy",
        "cpt_codes": ["81479"],
        "carc_codes": ["CO-50"],
        "content": (
            "Unlisted molecular pathology procedures and Whole Exome Panels (CPT "
            "81479) billed with general screening diagnosis code ICD-10 Z13.79 are "
            "denied under CARC CO-50 (Not deemed a medical necessity by the payer). "
            "To qualify for coverage on appeal, the ordering physician must submit: "
            "(1) a specific symptomatic or familial diagnostic ICD-10 code, (2) a "
            "signed Letter of Medical Necessity (LMN) with genetic counseling notes, "
            "and (3) for Platinum EPO Premier members, an Out-of-Network In-Network "
            "Exception (GAP exception) if no in-network reference lab performs the assay."
        ),
    },
    {
        "policy_id": "POL-LCD-L34636",
        "title": "Thoracic Diagnostic CT Imaging (CPT 71250) Prior Authorization & Clinical Attachment Policy",
        "category": "Radiology & Diagnostic Imaging Guidelines",
        "cpt_codes": ["71250", "94010"],
        "carc_codes": [],
        "content": (
            "Diagnostic Chest CT without contrast (CPT 71250) ordered for nonspecific "
            "abnormal lung findings (ICD-10 R91.8) or persistent asthma (ICD-10 J45.40) "
            "requires Prior Authorization. When a claim is in PENDING_REVIEW status "
            "awaiting PA clinical review (e.g., PA-2026-512), the ordering pulmonologist "
            "must upload the preceding chest radiograph (X-ray) report. Once attached, "
            "adjudication completes within 5 business days."
        ),
    },
]


# ============================================================================
# Deterministic Vector Embedding & Cosine Similarity Engine
# ============================================================================

_TOKEN_RE = re.compile(r"[a-z0-9_+-]{2,}")


def compute_sparse_embedding(text: str) -> dict[str, float]:
  """Computes a normalized L2 unit vector over unigrams and clinical code bigrams."""
  tokens = _TOKEN_RE.findall((text or "").lower())
  if not tokens:
    return {}
  features = list(tokens)
  for i in range(len(tokens) - 1):
    features.append(f"{tokens[i]}_{tokens[i + 1]}")
  counts = Counter(features)
  norm = math.sqrt(sum(float(c * c) for c in counts.values()))
  if norm == 0.0:
    return {}
  return {term: round(count / norm, 6) for term, count in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
  """Computes exact cosine similarity in [0.0, 1.0] between two L2-normalized vectors."""
  if not vec_a or not vec_b:
    return 0.0
  if len(vec_a) > len(vec_b):
    vec_a, vec_b = vec_b, vec_a
  dot = sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())
  return round(max(0.0, min(1.0, dot)), 4)


# ============================================================================
# Persistent SQLite Relational Database + Vector Store Repository
# ============================================================================


class ClaimsDatabaseAndVectorRepository:
  """ACID SQLite Relational Store & Clinical Policy Vector Store."""

  def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
    self.db_path = db_path
    self._lock = threading.RLock()
    self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._init_schema_and_seed()

  def _init_schema_and_seed(self) -> None:
    with self._lock, self._conn:
      self._conn.execute("PRAGMA journal_mode=WAL;")
      self._conn.execute(
          """
          CREATE TABLE IF NOT EXISTS members (
              member_id TEXT PRIMARY KEY,
              full_name TEXT NOT NULL,
              status TEXT NOT NULL,
              plan_name TEXT NOT NULL,
              payload_json TEXT NOT NULL
          )
          """
      )
      self._conn.execute(
          """
          CREATE TABLE IF NOT EXISTS claims (
              claim_id TEXT PRIMARY KEY,
              member_id TEXT NOT NULL,
              status TEXT NOT NULL,
              denial_code TEXT,
              total_billed REAL NOT NULL,
              payload_json TEXT NOT NULL,
              FOREIGN KEY (member_id) REFERENCES members(member_id)
          )
          """
      )
      self._conn.execute(
          """
          CREATE TABLE IF NOT EXISTS prior_authorizations (
              pa_id TEXT PRIMARY KEY,
              member_id TEXT NOT NULL,
              cpt_code TEXT NOT NULL,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL
          )
          """
      )
      self._conn.execute(
          """
          CREATE TABLE IF NOT EXISTS claim_appeals (
              appeal_id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL,
              member_id TEXT NOT NULL,
              status TEXT NOT NULL,
              submitted_timestamp TEXT NOT NULL,
              payload_json TEXT NOT NULL
          )
          """
      )
      self._conn.execute(
          """
          CREATE TABLE IF NOT EXISTS clinical_policy_vectors (
              policy_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              category TEXT NOT NULL,
              cpt_codes_json TEXT NOT NULL,
              carc_codes_json TEXT NOT NULL,
              content TEXT NOT NULL,
              embedding_json TEXT NOT NULL
          )
          """
      )
      self._conn.execute(
          """
          CREATE TABLE IF NOT EXISTS long_term_memories (
              memory_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              member_id TEXT,
              fact_summary TEXT NOT NULL,
              embedding_json TEXT NOT NULL,
              created_at TEXT NOT NULL
          )
          """
      )

      # Seed members
      for member_id, m in MOCK_MEMBERS.items():
        self._conn.execute(
            """
            INSERT OR REPLACE INTO members (member_id, full_name, status, plan_name, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                member_id,
                m["full_name"],
                m["status"],
                m["plan_name"],
                json.dumps(m),
            ),
        )

      # Seed claims
      for claim_id, c in MOCK_CLAIMS.items():
        self._conn.execute(
            """
            INSERT OR REPLACE INTO claims (claim_id, member_id, status, denial_code, total_billed, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id,
                c["member_id"],
                c["status"],
                c.get("denial_code"),
                float(c["total_billed"]),
                json.dumps(c),
            ),
        )

      # Seed prior authorizations
      for pa_id, pa in MOCK_PRIOR_AUTHS.items():
        self._conn.execute(
            """
            INSERT OR REPLACE INTO prior_authorizations (pa_id, member_id, cpt_code, status, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                pa_id,
                pa["member_id"],
                pa["cpt_code"],
                pa["status"],
                json.dumps(pa),
            ),
        )

      # Seed clinical policy vectors
      for doc in CLINICAL_POLICY_DOCUMENTS:
        index_text = (
            f"{doc['policy_id']} {doc['title']} {doc['category']} "
            f"{' '.join(doc['cpt_codes'])} {' '.join(doc['carc_codes'])} "
            f"{doc['content']}"
        )
        embedding = compute_sparse_embedding(index_text)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO clinical_policy_vectors
            (policy_id, title, category, cpt_codes_json, carc_codes_json, content, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["policy_id"],
                doc["title"],
                doc["category"],
                json.dumps(doc["cpt_codes"]),
                json.dumps(doc["carc_codes"]),
                doc["content"],
                json.dumps(embedding),
            ),
        )

  def find_member(self, query: str) -> Optional[MemberRecord]:
    q_norm = query.strip().upper()
    with self._lock:
      row = self._conn.execute(
          "SELECT payload_json FROM members WHERE UPPER(member_id) = ?",
          (q_norm,),
      ).fetchone()
      if row:
        return MemberRecord.model_validate(json.loads(row["payload_json"]))

      rows = self._conn.execute(
          "SELECT payload_json FROM members WHERE UPPER(full_name) LIKE ?",
          (f"%{q_norm}%",),
      ).fetchall()
      if rows:
        return MemberRecord.model_validate(json.loads(rows[0]["payload_json"]))
    return None

  def list_all_member_ids(self) -> list[str]:
    with self._lock:
      rows = self._conn.execute(
          "SELECT member_id FROM members ORDER BY member_id"
      ).fetchall()
      return [r["member_id"] for r in rows]

  def get_claims_for_member(
      self, member_id: str, status_filter: str = "ALL"
  ) -> list[ClaimRecord]:
    mid = member_id.strip().upper()
    sf = status_filter.strip().upper()
    with self._lock:
      if sf and sf != "ALL":
        rows = self._conn.execute(
            "SELECT payload_json FROM claims WHERE member_id = ? AND status = ? ORDER BY claim_id",
            (mid, sf),
        ).fetchall()
      else:
        rows = self._conn.execute(
            "SELECT payload_json FROM claims WHERE member_id = ? ORDER BY claim_id",
            (mid, ),
        ).fetchall()
      return [
          ClaimRecord.model_validate(json.loads(r["payload_json"]))
          for r in rows
      ]

  def get_claim(self, claim_id: str) -> Optional[ClaimRecord]:
    cid = claim_id.strip().upper()
    with self._lock:
      row = self._conn.execute(
          "SELECT payload_json FROM claims WHERE claim_id = ?", (cid,)
      ).fetchone()
      if row:
        return ClaimRecord.model_validate(json.loads(row["payload_json"]))
    return None

  def list_all_claim_ids(self) -> list[str]:
    with self._lock:
      rows = self._conn.execute(
          "SELECT claim_id FROM claims ORDER BY claim_id"
      ).fetchall()
      return [r["claim_id"] for r in rows]

  def query_prior_authorizations(
      self, member_id: str = "", pa_id: str = ""
  ) -> list[PriorAuthorizationRecord]:
    mid = member_id.strip().upper()
    pid = pa_id.strip().upper()
    with self._lock:
      if pid:
        rows = self._conn.execute(
            "SELECT payload_json FROM prior_authorizations WHERE pa_id = ?",
            (pid,),
        ).fetchall()
      elif mid:
        rows = self._conn.execute(
            "SELECT payload_json FROM prior_authorizations WHERE member_id = ? ORDER BY pa_id",
            (mid,),
        ).fetchall()
      else:
        rows = self._conn.execute(
            "SELECT payload_json FROM prior_authorizations ORDER BY pa_id"
        ).fetchall()
      return [
          PriorAuthorizationRecord.model_validate(json.loads(r["payload_json"]))
          for r in rows
      ]

  def save_appeal(self, appeal: AppealRecord) -> None:
    with self._lock, self._conn:
      self._conn.execute(
          """
          INSERT OR REPLACE INTO claim_appeals
          (appeal_id, claim_id, member_id, status, submitted_timestamp, payload_json)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (
              appeal.appeal_id,
              appeal.claim_id,
              appeal.member_id,
              appeal.status.value,
              appeal.submitted_timestamp,
              appeal.model_dump_json(),
          ),
      )

  def search_policy_vectors(
      self, query: str, top_k: int = 3
  ) -> list[PolicyDocumentMatch]:
    """Performs semantic cosine similarity search over the SQLite clinical policy vector store."""
    query_vec = compute_sparse_embedding(query)
    with self._lock:
      rows = self._conn.execute(
          "SELECT * FROM clinical_policy_vectors"
      ).fetchall()

    scored: list[PolicyDocumentMatch] = []
    for row in rows:
      doc_vec = json.loads(row["embedding_json"])
      sim = cosine_similarity(query_vec, doc_vec)
      # Boost exact code matches (CPT or CARC)
      q_upper = query.upper()
      cpts = json.loads(row["cpt_codes_json"])
      carcs = json.loads(row["carc_codes_json"])
      if any(code in q_upper for code in cpts) or any(
          code in q_upper for code in carcs
      ):
        sim = min(1.0, round(sim + 0.35, 4))

      scored.append(
          PolicyDocumentMatch(
              policy_id=row["policy_id"],
              title=row["title"],
              category=row["category"],
              cpt_codes=cpts,
              carc_codes=carcs,
              similarity_score=sim,
              content=row["content"],
          )
      )

    scored.sort(key=lambda item: item.similarity_score, reverse=True)
    return scored[:top_k]

  def insert_long_term_memory(
      self,
      *,
      memory_id: str,
      user_id: str,
      member_id: Optional[str],
      fact_summary: str,
      created_at: str,
  ) -> None:
    embedding = compute_sparse_embedding(fact_summary)
    with self._lock, self._conn:
      self._conn.execute(
          """
          INSERT OR REPLACE INTO long_term_memories
          (memory_id, user_id, member_id, fact_summary, embedding_json, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (
              memory_id,
              user_id,
              member_id,
              fact_summary,
              json.dumps(embedding),
              created_at,
          ),
      )

  def search_long_term_memories(
      self, query: str, user_id: Optional[str] = None, top_k: int = 3
  ) -> list[dict[str, Any]]:
    query_vec = compute_sparse_embedding(query)
    with self._lock:
      if user_id:
        rows = self._conn.execute(
            "SELECT * FROM long_term_memories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
      else:
        rows = self._conn.execute(
            "SELECT * FROM long_term_memories ORDER BY created_at DESC"
        ).fetchall()

    results = []
    for row in rows:
      mem_vec = json.loads(row["embedding_json"])
      sim = cosine_similarity(query_vec, mem_vec)
      results.append(
          {
              "memory_id": row["memory_id"],
              "user_id": row["user_id"],
              "member_id": row["member_id"],
              "fact_summary": row["fact_summary"],
              "similarity_score": sim,
              "created_at": row["created_at"],
          }
      )
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]


repository = ClaimsDatabaseAndVectorRepository()

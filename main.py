"""Healthcare Claims ADK Agent deployed on Vertex AI Agent Engine."""

import os
from typing import Any
from google.adk.agents import llm_agent
import vertexai
from vertexai.preview.reasoning_engines import AdkApp

# Ensure Vertex AI global config is initialized before constructing AdkApp
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "arnbtest")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ==============================================================================
# Mock Healthcare Claims Database
# ==============================================================================

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
        "plan_name": "Silver HMO Standard",
        "group_number": "GRP-44110",
        "status": "ACTIVE",
        "effective_date": "2026-01-01",
        "deductible_individual": 3000.00,
        "deductible_met": 650.00,
        "oop_max_individual": 7500.00,
        "oop_max_met": 980.00,
        "coinsurance_in_network": 0.30,
        "coinsurance_out_of_network": 1.00,  # HMO does not cover OON non-emergency
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
        "dob": "1992-07-22",
        "plan_name": "Platinum EPO Premier",
        "group_number": "GRP-99201",
        "status": "ACTIVE",
        "effective_date": "2026-01-01",
        "deductible_individual": 500.00,
        "deductible_met": 500.00,
        "oop_max_individual": 2500.00,
        "oop_max_met": 1420.00,
        "coinsurance_in_network": 0.10,
        "coinsurance_out_of_network": 1.00,
        "copays": {
            "pcp": 10.00,
            "specialist": 25.00,
            "urgent_care": 25.00,
            "emergency_room": 150.00,
        },
    },
    "MEM-1004": {
        "member_id": "MEM-1004",
        "full_name": "David Ross",
        "dob": "1964-01-19",
        "plan_name": "Bronze HDHP HSA",
        "group_number": "GRP-12009",
        "status": "INACTIVE",
        "effective_date": "2025-01-01",
        "termination_date": "2026-06-30",
        "deductible_individual": 6000.00,
        "deductible_met": 1800.00,
        "oop_max_individual": 8000.00,
        "oop_max_met": 1800.00,
        "coinsurance_in_network": 0.20,
        "coinsurance_out_of_network": 0.50,
        "copays": {},
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
                "icd10": "M23.211",
                "description": (
                    "Derangement of anterior horn of medial meniscus, right"
                    " knee"
                ),
            }
        ],
        "line_items": [
            {
                "cpt": "99214",
                "description": "Office/outpatient visit, established patient",
                "billed": 220.00,
                "allowed": 140.00,
                "plan_paid": 100.00,
                "patient_resp": 40.00,
                "notes": "Specialist copay ($40.00)",
            },
            {
                "cpt": "73721",
                "description": (
                    "MRI lower extremity joint (right knee) without contrast"
                ),
                "billed": 1450.00,
                "allowed": 980.00,
                "plan_paid": 764.00,
                "patient_resp": 216.00,
                "notes": "Deductible satisfied; 20% coinsurance applied",
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
                    "Derangement of anterior horn of medial meniscus, right"
                    " knee"
                ),
            }
        ],
        "line_items": [
            {
                "cpt": "29881",
                "description": (
                    "Arthroscopy, knee, surgical; with meniscectomy (medial OR"
                    " lateral)"
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
            "Precertification/authorization/notification absent. Note: Prior"
            " Authorization PA-2026-441 was approved for CPT 29881 under"
            " Pacific Orthopedic Associates (NPI-1234567890), but claim was"
            " billed under surgical facility NPI-1987654321."
        ),
        "appeal_eligible": True,
        "appeal_deadline": "2027-03-01",
    },
    "CLM-2026-9003": {
        "claim_id": "CLM-2026-9003",
        "member_id": "MEM-1002",
        "patient_name": "Michael Chen",
        "service_date": "2026-09-10",
        "received_date": "2026-09-11",
        "processed_date": "2026-09-15",
        "provider_name": "Metro Urgent Care Clinic",
        "provider_npi": "NPI-1122334455",
        "network_status": "IN_NETWORK",
        "diagnosis_codes": [
            {"icd10": "J02.9", "description": "Acute pharyngitis, unspecified"}
        ],
        "line_items": [
            {
                "cpt": "99203",
                "description": "Urgent care / office visit, new patient",
                "billed": 185.00,
                "allowed": 125.00,
                "plan_paid": 50.00,
                "patient_resp": 75.00,
                "notes": "Urgent care copay ($75.00)",
            },
            {
                "cpt": "87880",
                "description": "Strep A assay with optic readout",
                "billed": 45.00,
                "allowed": 35.00,
                "plan_paid": 35.00,
                "patient_resp": 0.00,
                "notes": "Bundled diagnostic lab covered at 100%",
            },
        ],
        "total_billed": 230.00,
        "total_allowed": 160.00,
        "total_plan_paid": 85.00,
        "total_patient_responsibility": 75.00,
        "status": "PAID",
        "eob_number": "EOB-90104",
        "denial_code": None,
        "denial_reason": None,
        "appeal_eligible": False,
    },
    "CLM-2026-9004": {
        "claim_id": "CLM-2026-9004",
        "member_id": "MEM-1002",
        "patient_name": "Michael Chen",
        "service_date": "2026-09-18",
        "received_date": "2026-09-19",
        "processed_date": "2026-09-23",
        "provider_name": "Apex Cardiology Group",
        "provider_npi": "NPI-1556677889",
        "network_status": "OUT_OF_NETWORK",
        "diagnosis_codes": [{"icd10": "R00.2", "description": "Palpitations"}],
        "line_items": [
            {
                "cpt": "93306",
                "description": (
                    "Echocardiography, transthoracic, real-time with image"
                    " documentation (2D), complete"
                ),
                "billed": 1950.00,
                "allowed": 0.00,
                "plan_paid": 0.00,
                "patient_resp": 1950.00,
                "notes": "Denied: Out-of-network provider on HMO plan",
            }
        ],
        "total_billed": 1950.00,
        "total_allowed": 0.00,
        "total_plan_paid": 0.00,
        "total_patient_responsibility": 1950.00,
        "status": "DENIED",
        "eob_number": "EOB-90881",
        "denial_code": "CO-242",
        "denial_reason": (
            "Services not provided by network/primary care providers. Silver"
            " HMO Standard requires an approved out-of-network referral waiver"
            " for non-emergency specialist services."
        ),
        "appeal_eligible": True,
        "appeal_deadline": "2027-03-17",
    },
    "CLM-2026-9005": {
        "claim_id": "CLM-2026-9005",
        "member_id": "MEM-1003",
        "patient_name": "Elena Rodriguez",
        "service_date": "2026-09-20",
        "received_date": "2026-09-21",
        "processed_date": None,
        "provider_name": "Golden Gate Women's Health",
        "provider_npi": "NPI-1443322110",
        "network_status": "IN_NETWORK",
        "diagnosis_codes": [
            {
                "icd10": "Z00.00",
                "description": (
                    "Encounter for general adult medical examination without"
                    " abnormal findings"
                ),
            }
        ],
        "line_items": [
            {
                "cpt": "99213",
                "description": "Office/outpatient visit, established patient",
                "billed": 160.00,
                "allowed": 120.00,
                "plan_paid": 0.00,
                "patient_resp": 0.00,
                "notes": (
                    "Pending adjudication (Preventive wellness visit covered"
                    " 100%)"
                ),
            },
            {
                "cpt": "80053",
                "description": "Comprehensive metabolic panel",
                "billed": 95.00,
                "allowed": 70.00,
                "plan_paid": 0.00,
                "patient_resp": 0.00,
                "notes": "Pending adjudication",
            },
        ],
        "total_billed": 255.00,
        "total_allowed": 190.00,
        "total_plan_paid": 0.00,
        "total_patient_responsibility": 0.00,
        "status": "PENDING_REVIEW",
        "eob_number": None,
        "denial_code": None,
        "denial_reason": None,
        "appeal_eligible": False,
        "estimated_completion_date": "2026-09-28",
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
            "Approved for 1 surgical procedure. Facility NPI can be updated to"
            " Bay Area Surgical Center (NPI-1987654321) via claim appeal or"
            " retro-auth request."
        ),
    },
    "PA-2026-512": {
        "pa_id": "PA-2026-512",
        "member_id": "MEM-1003",
        "patient_name": "Elena Rodriguez",
        "cpt_code": "70553",
        "procedure_description": "MRI Brain with and without contrast",
        "status": "PENDING_CLINICAL_INFO",
        "approved_provider": "UCSF Imaging Center",
        "approved_npi": "NPI-1778899001",
        "valid_from": None,
        "valid_to": None,
        "notes": (
            "Awaiting clinical documentation showing 4 weeks of conservative"
            " management from ordering physician."
        ),
    },
}


# ==============================================================================
# ADK Tool Definitions
# ==============================================================================


def get_member_eligibility(member_id_or_name: str) -> dict[str, Any]:
  """Look up a member's insurance eligibility, plan benefits, deductible, and out-of-pocket maximums.

  Args:
    member_id_or_name: The member ID (e.g. 'MEM-1001') or full/partial name
      (e.g. 'Sarah Jenkins').

  Returns:
    A dictionary containing member eligibility and benefit accumulation details.
  """
  query = member_id_or_name.strip().upper()
  if query in MOCK_MEMBERS:
    return {"found": True, "member": MOCK_MEMBERS[query]}

  matches = [
      m
      for m in MOCK_MEMBERS.values()
      if member_id_or_name.strip().lower() in m["full_name"].lower()
  ]
  if matches:
    return {"found": True, "member": matches[0]}

  return {
      "found": False,
      "error": f"No member found matching '{member_id_or_name}'.",
      "available_members": [
          {"member_id": k, "full_name": v["full_name"], "plan": v["plan_name"]}
          for k, v in MOCK_MEMBERS.items()
      ],
  }


def list_member_claims(
    member_id: str, status_filter: str = ""
) -> dict[str, Any]:
  """List healthcare claims for a given member ID, optionally filtered by claim status.

  Args:
    member_id: The member ID (e.g. 'MEM-1001', 'MEM-1002', 'MEM-1003'). If
      'ALL' is passed, returns all mock claims across all members.
    status_filter: Optional claim status filter such as 'PAID', 'DENIED', or
      'PENDING_REVIEW'. Leave empty for all statuses.

  Returns:
    A dictionary with matching claim summaries.
  """
  mid = member_id.strip().upper()
  status_norm = status_filter.strip().upper()

  claims = [
      c
      for c in MOCK_CLAIMS.values()
      if (mid == "ALL" or c["member_id"] == mid)
      and (not status_norm or c["status"] == status_norm)
  ]
  return {
      "member_id": mid,
      "status_filter": status_norm or "ALL",
      "count": len(claims),
      "claims": [
          {
              "claim_id": c["claim_id"],
              "member_id": c["member_id"],
              "patient_name": c["patient_name"],
              "service_date": c["service_date"],
              "provider_name": c["provider_name"],
              "total_billed": c["total_billed"],
              "total_plan_paid": c["total_plan_paid"],
              "total_patient_responsibility": c["total_patient_responsibility"],
              "status": c["status"],
              "denial_code": c["denial_code"],
          }
          for c in claims
      ],
  }


def get_claim_details(claim_id: str) -> dict[str, Any]:
  """Retrieve full adjudication details, CPT line items, ICD-10 diagnosis codes, and denial reasons for a specific claim.

  Args:
    claim_id: The claim identifier (e.g. 'CLM-2026-9001', 'CLM-2026-9002').

  Returns:
    Detailed claim information dictionary.
  """
  cid = claim_id.strip().upper()
  if cid in MOCK_CLAIMS:
    return {"found": True, "claim": MOCK_CLAIMS[cid]}
  return {
      "found": False,
      "error": f"Claim '{claim_id}' not found.",
      "available_claim_ids": list(MOCK_CLAIMS.keys()),
  }


def check_prior_authorization(
    member_id: str = "", pa_id: str = ""
) -> dict[str, Any]:
  """Look up prior authorization (PA) records by PA ID or Member ID.

  Args:
    member_id: Optional member ID (e.g. 'MEM-1001').
    pa_id: Optional prior authorization ID (e.g. 'PA-2026-441').

  Returns:
    Matching prior authorization records.
  """
  pid = pa_id.strip().upper()
  mid = member_id.strip().upper()

  if pid and pid in MOCK_PRIOR_AUTHS:
    return {"found": True, "prior_authorizations": [MOCK_PRIOR_AUTHS[pid]]}

  results = [
      pa
      for pa in MOCK_PRIOR_AUTHS.values()
      if (not mid or pa["member_id"] == mid)
  ]
  return {
      "found": bool(results),
      "count": len(results),
      "prior_authorizations": results,
  }


def estimate_patient_responsibility(
    member_id: str,
    cpt_code: str,
    billed_amount: float,
    in_network: bool = True,
) -> dict[str, Any]:
  """Calculate estimated allowed amount, deductible application, coinsurance, and patient responsibility for a procedure.

  Args:
    member_id: The member ID (e.g. 'MEM-1001').
    cpt_code: The CPT procedure code (e.g. '29881', '73721', '99214').
    billed_amount: Provider billed amount in USD.
    in_network: True if the provider is in-network, False if out-of-network.

  Returns:
    Cost breakdown estimate for the member.
  """
  mid = member_id.strip().upper()
  if mid not in MOCK_MEMBERS:
    return {"error": f"Member '{member_id}' not found."}

  member = MOCK_MEMBERS[mid]
  if member["status"] != "ACTIVE":
    return {
        "member_id": mid,
        "status": member["status"],
        "estimated_plan_paid": 0.0,
        "estimated_patient_responsibility": billed_amount,
        "note": "Member coverage is INACTIVE; plan pays $0.00.",
    }

  if not in_network and member["coinsurance_out_of_network"] >= 1.0:
    return {
        "member_id": mid,
        "plan_name": member["plan_name"],
        "in_network": False,
        "estimated_allowed_amount": 0.0,
        "estimated_plan_paid": 0.0,
        "estimated_patient_responsibility": billed_amount,
        "note": (
            f"{member['plan_name']} does not cover out-of-network non-emergency"
            " services."
        ),
    }

  # Standard mock fee schedule discount: 70% of billed for in-network
  allowed = round(billed_amount * (0.70 if in_network else 0.50), 2)
  remaining_deductible = max(
      0.0, member["deductible_individual"] - member["deductible_met"]
  )
  remaining_oop = max(
      0.0, member["oop_max_individual"] - member["oop_max_met"]
  )

  applied_to_deductible = min(allowed, remaining_deductible)
  after_deductible = allowed - applied_to_deductible
  coinsurance_rate = (
      member["coinsurance_in_network"]
      if in_network
      else member["coinsurance_out_of_network"]
  )
  coinsurance_amount = round(after_deductible * coinsurance_rate, 2)

  patient_resp = min(
      remaining_oop, round(applied_to_deductible + coinsurance_amount, 2)
  )
  plan_paid = round(allowed - patient_resp, 2)

  return {
      "member_id": mid,
      "plan_name": member["plan_name"],
      "cpt_code": cpt_code,
      "in_network": in_network,
      "billed_amount": billed_amount,
      "estimated_allowed_amount": allowed,
      "applied_to_deductible": applied_to_deductible,
      "coinsurance_rate": coinsurance_rate,
      "coinsurance_amount": coinsurance_amount,
      "estimated_patient_responsibility": patient_resp,
      "estimated_plan_paid": plan_paid,
  }


def submit_claim_appeal(
    claim_id: str,
    appeal_reason: str,
    supporting_reference: str = "",
) -> dict[str, Any]:
  """Submit a formal appeal or retroactive prior-authorization correction for a denied healthcare claim.

  Args:
    claim_id: The denied claim ID (e.g. 'CLM-2026-9002').
    appeal_reason: Explanation for why the claim should be reprocessed/approved.
    supporting_reference: Optional supporting PA number or clinical reference
      (e.g. 'PA-2026-441').

  Returns:
    Confirmation of the submitted appeal and updated claim status.
  """
  cid = claim_id.strip().upper()
  if cid not in MOCK_CLAIMS:
    return {"success": False, "error": f"Claim '{claim_id}' not found."}

  claim = MOCK_CLAIMS[cid]
  if claim["status"] != "DENIED":
    return {
        "success": False,
        "claim_id": cid,
        "current_status": claim["status"],
        "error": (
            f"Only DENIED claims can be appealed (current status:"
            f" {claim['status']})."
        ),
    }

  appeal_id = f"APL-2026-{cid.split('-')[-1]}"
  claim["status"] = "APPEAL_SUBMITTED"
  claim["appeal_id"] = appeal_id
  claim["appeal_reason"] = appeal_reason
  claim["supporting_reference"] = supporting_reference

  return {
      "success": True,
      "appeal_id": appeal_id,
      "claim_id": cid,
      "member_id": claim["member_id"],
      "new_status": "APPEAL_SUBMITTED",
      "appeal_reason": appeal_reason,
      "supporting_reference": supporting_reference,
      "estimated_resolution_days": 5,
      "message": (
          f"Appeal {appeal_id} submitted for claim {cid}. Status updated to"
          " APPEAL_SUBMITTED."
      ),
  }


# ==============================================================================
# ADK Root Agent & AdkApp Definition
# ==============================================================================

AGENT_INSTRUCTION = """You are the Healthcare Claims Adjudication & Member Support Agent.
You assist members, healthcare providers, and claims adjusters with:
1. Checking member eligibility, plan benefits, deductibles, copays, and out-of-pocket maximums (`get_member_eligibility`).
2. Listing member claims and checking adjudication statuses (`list_member_claims`).
3. Investigating specific claim line items, CPT/HCPCS procedures, ICD-10 diagnosis codes, EOBs, and denial codes (`get_claim_details`).
4. Verifying prior authorizations and identifying NPI or coverage mismatches (`check_prior_authorization`).
5. Estimating patient out-of-pocket responsibility for procedures (`estimate_patient_responsibility`).
6. Submitting claim appeals or prior-authorization corrections for denied claims (`submit_claim_appeal`).

Always use your tools to retrieve exact numbers, dates, CPT/ICD-10 codes, and denial reasons before answering.
Present financial breakdowns clearly (Billed Amount, Allowed Amount, Plan Paid, and Patient Responsibility) and suggest actionable next steps when a claim is denied."""

root_agent = llm_agent.LlmAgent(
    name="Healthcare_Claims_Agent",
    model="gemini-2.5-flash",
    description=(
        "ADK Healthcare Claims Agent for member eligibility verification, claim"
        " status lookup, denial root-cause analysis, prior authorization"
        " checks, cost estimation, and appeals."
    ),
    instruction=AGENT_INSTRUCTION,
    tools=[
        get_member_eligibility,
        list_member_claims,
        get_claim_details,
        check_prior_authorization,
        estimate_patient_responsibility,
        submit_claim_appeal,
    ],
)

app = AdkApp(agent=root_agent)

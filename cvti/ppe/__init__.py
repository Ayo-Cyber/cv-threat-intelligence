"""PPE compliance: per-person, three-state, policy-driven.

Not a "compliance percentage". A person is `violation` (a required item is
confidently absent), `compliant` (every required item confidently present and
worn), or `unable` (something required could not be assessed — feet cut off,
ownership of a vest unclear between two overlapping people). Failure to see
an item is never proof it is missing.

    policy.py   what the CUSTOMER requires, per zone (config, not the model)
    assess.py   detect → associate with the right person → three states →
                evidence over time → verdict
    scanner.py  the bounded worker: off the camera path, ~1 Hz per camera,
                drops stale work instead of queueing it, reports coverage
"""
from cvti.ppe.assess import Compliance, TrackEvidence, assess_compliance, observe_people
from cvti.ppe.policy import CATALOG, PPEItem, PPEPolicy, load_ppe_policy

__all__ = ["CATALOG", "Compliance", "PPEItem", "PPEPolicy", "TrackEvidence",
           "assess_compliance", "load_ppe_policy", "observe_people"]

```
KPI SCORECARD — measured on the frozen manifest (digest e56b42776bd164b8)
 SN  KPI                         target  measured          wilson     n  verdict
  8  Theft Detection              >=90%     21.9%       [16%,29%]   155  NOT MET
  9  Suspicious Activity          >=90%     48.9%       [42%,56%]   174  NOT MET

A row is MET only when its CONSERVATIVE Wilson bound clears the target; SMOKE rows need more clips before anyone signs them (tools/kpi_manifest.py status says where to get them).
```

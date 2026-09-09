```
KPI SCORECARD — measured on the frozen manifest (digest e56b42776bd164b8)
 SN  KPI                         target  measured          wilson     n  verdict
  2  False Positive Rate           <=5%     30.0%       [18%,45%]    40  BAKEOFF SUBSET (n=40)
  8  Theft Detection              >=90%     55.0%       [40%,69%]    40  BAKEOFF SUBSET (n=40)
  9  Suspicious Activity          >=90%     90.0%       [77%,96%]    40  BAKEOFF SUBSET (n=40)

A row is MET only when its CONSERVATIVE Wilson bound clears the target; SMOKE rows need more clips before anyone signs them (tools/kpi_manifest.py status says where to get them).
```

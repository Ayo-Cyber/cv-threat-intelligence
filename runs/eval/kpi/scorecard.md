```
KPI SCORECARD — measured on the frozen manifest (digest e56b42776bd164b8)
 SN  KPI                         target  measured          wilson     n  verdict
  2  False Positive Rate           <=5%      4.7%         [2%,9%]   170  NOT MET
  8  Theft Detection              >=90%     17.4%       [13%,23%]   224  NOT MET
  9  Suspicious Activity          >=90%     62.9%       [53%,71%]   105  NOT MET

A row is MET only when its CONSERVATIVE Wilson bound clears the target; SMOKE rows need more clips before anyone signs them (tools/kpi_manifest.py status says where to get them).
```

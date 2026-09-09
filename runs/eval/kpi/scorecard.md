```
KPI SCORECARD — measured on the frozen manifest (digest bb6f7f3fe154be5c)
 SN  KPI                         target  measured          wilson     n  verdict
  2  False Positive Rate           <=5%      4.7%         [2%,9%]   170  NOT MET
  5  Person Detection             >=95%     73.5%       [64%,81%]   102  NOT MET
  6  Intrusion Detection          >=95%     59.4%       [52%,67%]   165  NOT MET
  7  Loitering Detection          >=95%     27.3%       [10%,57%]    11  SMOKE (n too small)
  8  Theft Detection              >=90%     21.9%       [16%,29%]   155  NOT MET
  9  Suspicious Activity          >=90%     48.5%       [41%,56%]   173  NOT MET

A row is MET only when its CONSERVATIVE Wilson bound clears the target; SMOKE rows need more clips before anyone signs them (tools/kpi_manifest.py status says where to get them).
```

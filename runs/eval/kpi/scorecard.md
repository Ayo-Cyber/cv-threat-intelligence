```
MOCK GATE — wiring check only; nothing here is a signable number
KPI SCORECARD — measured on the frozen manifest (digest 85162b724bbaa223)
 SN  KPI                         target  measured          wilson     n  verdict
  2  False Positive Rate           <=5%     25.9%       [20%,33%]   170  NOT MET
  5  Person Detection             >=95%     92.2%       [85%,96%]   103  NOT MET
  6  Intrusion Detection          >=95%     90.3%       [85%,94%]   165  NOT MET
  7  Loitering Detection          >=95%     45.5%       [21%,72%]    11  SMOKE (n too small)
  8  Theft Detection              >=90%     42.6%       [35%,50%]   155  NOT MET
  9  Suspicious Activity          >=90%     97.1%       [93%,99%]   173  MET

A row is MET only when its CONSERVATIVE Wilson bound clears the target; SMOKE rows need more clips before anyone signs them (tools/kpi_manifest.py status says where to get them).
```

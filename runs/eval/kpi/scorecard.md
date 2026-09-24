```
MOCK GATE — wiring check only; nothing here is a signable number
KPI SCORECARD — measured on the frozen manifest (digest 047e113c08380eb1)
 SN  KPI                         target  measured          wilson     n  verdict
  2  False Positive Rate           <=5%     25.9%       [20%,33%]   170  NOT MET
  5  Person Detection             >=95%     92.5%       [86%,96%]   107  NOT MET
  6  Intrusion Detection          >=95%     90.3%       [85%,94%]   165  NOT MET
  7  Loitering Detection          >=95%     27.8%       [12%,51%]    18  SMOKE (n too small)
  8  Theft Detection              >=90%     46.9%       [39%,55%]   162  NOT MET
  9  Suspicious Activity          >=90%     97.2%       [94%,99%]   178  MET

A row is MET only when its CONSERVATIVE Wilson bound clears the target; SMOKE rows need more clips before anyone signs them (tools/kpi_manifest.py status says where to get them).
```

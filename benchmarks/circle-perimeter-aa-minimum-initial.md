# Anti-aliased circle: initial one-thread comparison

Initial descriptive allocation-inclusive public-call comparison, not an optimization KEEP decision or whole-library speed claim; every loss retained.

| Case | Original ms | C++ ms | Original/current |
|---|---:|---:|---:|
| circle-aa/tiny | 0.004616909 | 0.001820918 | 2.535483958x |
| circle-aa/zero | 0.003338163 | 0.001769071 | 1.886958738x |
| circle-aa/cache | 0.035381522 | 0.002661878 | 13.291939169x |
| circle-aa/long | 4.561367259 | 0.103263039 | 44.172312586x |
| circle-aa/clipped | 0.248164045 | 0.007436302 | 33.371971955x |
| circle-aa/clipped-away | 0.131573240 | 0.004071371 | 32.316693085x |
| circle-aa/translated | 0.125471522 | 0.004973921 | 25.225876656x |
| circle-aa/negative | 0.003307922 | 0.001782057 | 1.856237865x |

# RegionProperties partial state: initial one-thread comparison

Twelve one-thread cases for the partial distinct-subclass RegionProperties bridge: constructor versus constructor+sevenproperties, allgains/losses retained.7observed/37unassessed canonical properties; no wholeclass/libraryspeed orKEEPgate claim.

| Case | Original ms | Current ms | Original/current |
|---|---:|---:|---:|
| regionprops-state/tiny2d-constructor | 0.023683089 | 0.026439238 | 0.895755362x |
| regionprops-state/tiny2d-properties | 0.088523935 | 0.076996047 | 1.149720514x |
| regionprops-state/medium2d-constructor-sparse | 0.123304974 | 0.216730198 | 0.568933058x |
| regionprops-state/medium2d-properties-cache-on | 0.549484597 | 0.958349692 | 0.573365444x |
| regionprops-state/medium2d-transpose-cache-off | 0.669388366 | 6.084659579 | 0.110012460x |
| regionprops-state/large2d-c-cache-off | 6.829987471 | 13.840654444 | 0.493472870x |
| regionprops-state/large2d-f-coords | 5.374666579 | 27.546868200 | 0.195109896x |
| regionprops-state/medium2d-reverse-sparse | 2.189494019 | 4.585288200 | 0.477504123x |
| regionprops-state/small3d-constructor | 0.052225012 | 0.063447113 | 0.823126691x |
| regionprops-state/medium3d-f-cache-off | 0.767663527 | 10.004331333 | 0.076733117x |
| regionprops-state/large3d-cache-off | 6.306104737 | 14.333662444 | 0.439950694x |
| regionprops-state/large3d-sparse-spacing-offset | 4.970426917 | 12.484317900 | 0.398133639x |

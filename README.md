# pv-system-model

This is a model for PV plants/systems based on pvlib [1]. One relatively novel feature is that it allows for row-to-row shade, using many concepts from [2], with the addition of cross-axis slope while still allowing for bifacial modules.

The documentation needs improvement, and is a work in progress.

See [quick_demo.ipynb](quick_demo.ipynb) and https://github.com/williamhobbs/2025-pvrw-trackers for some example uses. 

[1] Anderson, K., Hansen, C., Holmgren, W., Jensen, A., Mikofski, M., and Driesse, A. “pvlib python: 2023 project update.” Journal of Open Source Software, 8(92), 5994, (2023). DOI: [http://dx.doi.org/10.21105/joss.05994](10.21105/joss.05994).

[2] Hobbs, W., Anderson, K., Mikofski, M., and Ghiz, M. "An approach to modeling linear and non-linear self-shading losses with pvlib." 2024 PV Performance Modeling Collaborative (PVPMC). https://github.com/williamhobbs/2024_pvpmc_self_shade 
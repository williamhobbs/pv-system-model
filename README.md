# pv-system-model

This is a model for PV plants/systems based on pvlib [1]. One relatively novel feature is that it allows for row-to-row shade, using many concepts from [2], with the addition of cross-axis slope while still allowing for bifacial modules.

## Features
### Common modeling steps
The `model_pv_power` function in `pv_model.py` combines many common modeling steps, including:
- solar position
- surface tilt and azimuth for a number of tracking options (or fixed tilt):
  - standard backtracking
  - slope-aware backtracking
  - above options, modified with custom gcr settings (a single value or separate morning/afternoon values)
  - fully custom tilt/azimuth timeseries (calculated separately by the user and passed to the function)
- irradiance transposition with a selectable model (default of perez-driesse)
  - optionally, measured POA data can be used instead
- spectral modifier adjustment with `pvlib.spectrum.spectral_factor_firstsolar` if wanted
- incident angle modifier (iam) adjustment
- cell temperature calculation with `pvlib.temperature.faiman`
- dc power with `pvlib.pvsystem.pvwatts_dc`
  - including a user-defined dc loss fraction
- ac power with `pvlib.inverter.pvwatts`

### Less common features
It also includes less common features that may be useful, including:
- separate rear-side irradiance with `pvlib.bifacial.infinite_sheds.get_irradiance` for bifacial system
  - `infinite_sheds` is only used for the rear to allow for cross-axis slope to be included in front-side irradiance calculations, as `infinite_sheds` currently only works with flat terrain.
- shade loss calculations for thin-film and common crystalline modules in portrait orientation
  - linear (thin-film) and non-linear (crystalline) shade losses
  - crystalline modules can be either with older-style square cells and modern modern "twin" modules with half-cut cells
  - allows for multiple "courses" of modules, e.g., "2P" where modules are in portrat two-high up the racking
  - *Note: Crystalline modules in landscape are not an option currently. Anecdotally, there are vary few utility-scale fixed tilt solar projects in the US with crystalline modules in landscape orientation. I'm less certain about tracking projects with landscape crystalline, but those would likely use backtracking, in which case shade losses are not typically a focus area.*

## Inputs and outputs
The `model_pv_power` function takes in:
- a resource data DataFrame, similar to the format that is returned by `pvlib.iotools.get*` functions
- many plant/system specifications (which can also be passed in as a Python dictionary) that follow conventions and definitions in https://github.com/williamhobbs/pv-plant-specifications, specifically [pv-plant-specification-rev5.csv](https://github.com/williamhobbs/pv-plant-specifications/blob/main/pv-plant-specification-rev5.csv)

The function returns:
- an ac power timeseries
- a modified version of the resource data DataFrame that includes the addition of modeled POA irradiance and modeled cell temperature. This might be useful for comparison with measured values from operating plants for performance engineering purposes.

## Examples

See [quick_demo.ipynb](quick_demo.ipynb) and https://github.com/williamhobbs/2025-pvrw-trackers for some example uses. It is also used in:
 - https://github.com/williamhobbs/energy-forecasting-tools
 - https://github.com/williamhobbs/PVPMC_2025
 - https://github.com/williamhobbs/PVSC-2025-daily-energy-forecaster

## References

[1] Anderson, K., Hansen, C., Holmgren, W., Jensen, A., Mikofski, M., and Driesse, A. “pvlib python: 2023 project update.” Journal of Open Source Software, 8(92), 5994, (2023). DOI: [http://dx.doi.org/10.21105/joss.05994](10.21105/joss.05994).

[2] Hobbs, W., Anderson, K., Mikofski, M., and Ghiz, M. "An approach to modeling linear and non-linear self-shading losses with pvlib." 2024 PV Performance Modeling Collaborative (PVPMC). https://github.com/williamhobbs/2024_pvpmc_self_shade 

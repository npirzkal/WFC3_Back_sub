# WFC3_Back_Sub
This is my personal version of a code that implements the multi-components HST grism background subtraction descrived in  Pirzkal et al. 2017 (https://ui.adsabs.harvard.edu/abs/2017ApJ...846...84P/abstract) and Pirzkal & Ryan, 2020 (https://ui.adsabs.harvard.edu/abs/2020wfc..rept....4P/abstract).

# Sample Usage: #
Typical use, to produce CALWF3 calibrated data which are background subtracted, starting from RAW files:
```
import WFC3_Back_Sub
visits = WFC3_Back_Sub.get_visits("*_raw.fits")
for obsid in visits[gfilter]:
	t = WFC3_Back_Sub.Sub_Back(visits[gfilter][obsid],gfilter)
	t.process_obs_ids()
```
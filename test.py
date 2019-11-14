import WFC3_Back_Sub

obs_id = "icoi3na5q"
WFC3_Back_Sub.get_data(obs_id)

obs_ids = WFC3_Back_Sub.get_visits("icoi3na5q_raw.fits")

print(obs_ids)

import WFC3_Back_Sub
visits = WFC3_Back_Sub.get_visits("*_raw.fits")
gfilter = "G102"
for obsid in visits[gfilter]:
	t = WFC3_Back_Sub.Sub_Back(visits[gfilter]["icoi3n"],gfilter)
	t.process_obs_ids()
	t.restore_FF()
	t.diagnostic_plots()


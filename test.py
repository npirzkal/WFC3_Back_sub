import WFC3_Back_Sub

obs_id = "icoi3na5q"
WFC3_Back_Sub.get_data(obs_id)

obs_ids = WFC3_Back_Sub.get_visits("*_raw.fits")

print(obs_ids)

WFC3_Back_Sub.process_obs_ids(obs_ids["G102"]["icoi3n"])






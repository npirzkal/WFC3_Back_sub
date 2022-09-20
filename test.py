import WFC3_Back_Sub

obs_id = "idxjeaciq"

# import os,urllib,shutil
# url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/icoi3n030_asn.fits"

# raw_name = "./icoi3n030_asn.fits"

# if os.path.isfile(raw_name):
#     os.unlink(raw_name)
    
# with urllib.request.urlopen(url) as response, open(raw_name, 'wb') as out_file:
#     shutil.copyfileobj(response, out_file)




WFC3_Back_Sub.get_data(obs_id)

obs_ids = WFC3_Back_Sub.get_visits("idxjeaciq_raw.fits")

print(obs_ids)

import WFC3_Back_Sub
visits = WFC3_Back_Sub.get_visits("*_raw.fits")
print(visits)
gfilter = "G141"
for obsid in visits[gfilter]:
	t = WFC3_Back_Sub.Sub_Back(visits[gfilter]["idxjea"],gfilter)
	t.process_obs_ids()
	t.restore_FF()
	t.diagnostic_plots()


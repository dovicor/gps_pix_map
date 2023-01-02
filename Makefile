
all: test zzz

test tests: test1.html test2.html test99.html


zzz: DvoTahoe2022.html



install: test1.html test2.html DvoTahoe2022.html
	cp test1.html /var/www/html/dvo/test
	cp test2.html /var/www/html/dvo/test
	cp DvoTahoe2022.html /var/www/html/dvo/DvoTahoe2022.html

clean:
	-rm test1_x.html
	-rm test1.html
	-rm test2_x.html
	-rm test2.html
	-rm test99_x.html
	-rm test99.html
	-rm dvoTahoe2022.html
	-rm DvoTahoe2022.html

test1.html:
	time python3 ./gps_pix_map.py \
		-outfile test1_x.html \
		-tracks /home/don/GPX/Tracks/TahoeRimTrail2022 "Don's Hiking Tahoe Rim Trail (orange)" "{'color': 'orange', 'weight': 4, 'opacity': 0.5}" \
		-tracks /home/don/GPX/Tracks/HikingTahoe2022 "Don's Hiking Tahoe (brown)" "{'color': 'brown',  'weight': 4, 'opacity': 0.5}" \
		-tracks /home/don/GPX/Tracks/HikingKokanee2022 "Don's Hiking from Kokanee (red)" "{'color': 'red',    'weight': 4, 'opacity': 0.5}" \
		-tracks /home/don/GPX/Tracks/KayakTahoe2022 "Don's Kayaking (blue)" "{'color': 'blue',   'weight': 4, 'opacity': 0.5}" \
		-photos /home/don/Pictures/tahoe_hike_2022_q1 /dvo/images "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-photos /home/don/Pictures/tahoe_hike_2022_q2 /dvo/images "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-photos /home/don/Pictures/tahoe_hike_2022_q3 /dvo/images "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-photos /home/don/Pictures/tahoe_hike_2022_q4 /dvo/images "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-fixed /home/don/GPX/Routes/trt-500pointsPerTrack.gpx "Tahoe Rim Trail (yellow)" "{'color':'yellow',  'weight': 2}" '<a href="http://www.tahoerimtrail.org">Tahoe Rim Trail</a>' \
		-pm_group 200 \
		-polygon /home/don/GPX/SkiAreas/Heavenly.gpx "Heavenly Mountain Resort" "{'color': 'purple', 'dash_array': '16 8'}" '<a href="http://www.skiheavenly.com">Heavenly Mountain Resort<\a>' \
		-skilifts /home/don/GPX/SkiAreas/HeavenlyLifts.csv "Heavenly Lifts" "{'color': 'black',  'weight':2, 'opacity':0.75, 'dash_array': '14 4'}" \
		-showbbox 2 \
		-cachecorrections ~/.gps_pix_map/jpeg_gps_corrections.csv \
		-include 39 -120 50 \
		-exclude 38.92828 -120.02159 38.93360 -120.00747 \
		-exclude 38.92928 -120.01763 38.93602 -120.00478 \
		-checkbox_how day \
		-ss_url_prefix https://kokanee.mynetgear.com/dvo/images \
		-profile \
		-debug 1
	sed -e 's/"dashArray": "class_/"className": "/' < test1_x.html > test1.html


test2.html:
	time python3 ./gps_pix_map.py \
		-out test2_x.html \
		-skitracks /home/don/GPX/Tracks/SkiingTahoe_22-23 "DVO Ski tracks" "{'color': 'pink',  'weight':2, 'opacity':0.5}" "Heavenly Mountain Resort" \
		-polygon /home/don/GPX/SkiAreas/Heavenly.gpx "Heavenly Mountain Resort" "{'color': 'purple', 'dash_array': '16 8'}" '<a href="http://www.skiheavenly.com">Heavenly Mountain Resort<\a>' \
		-skilifts /home/don/GPX/SkiAreas/HeavenlyLifts.csv "Heavenly Lifts" "{'color': 'black',  'weight':2, 'opacity':0.75, 'dash_array': '14 4'}" "Heavenly Mountain Resort" \
		-skirun /home/don/GPX/SkiAreas/HeavenlyRuns.csv "Heavenly Runs" "{'color': 'olive',  'weight':2, 'opacity':0.5}" "Heavenly Mountain Resort" \
		-photos /home/don/Pictures/tahoe_ski_2022_q4 /dvo/images "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-pm_group 200 \
		-profile \
		-ss_url_prefix https://kokanee.mynetgear.com/dvo/images \
		-checkbox_how day \
		-debug 1
	sed -e 's/"dashArray": "class_/"className": "/' < test2_x.html > test2.html



test99.html:
	time python3 ./gps_pix_map.py \
		-out test99_x.html \
		-skitracks /home/don/GPX/Tracks/SkiingTahoe_22-23/20221216_Heavenly.gpx "DVO Ski tracks" "{'color': 'pink',  'weight':2, 'opacity':0.5}" "Heavenly Mountain Resort" \
		-polygon /home/don/GPX/SkiAreas/Heavenly.gpx "Heavenly Mountain Resort" "{'color': 'purple', 'dash_array': '16 8'}" '<a href="http://www.skiheavenly.com">Heavenly Mountain Resort<\a>' \
		-skilifts /home/don/GPX/SkiAreas/HeavenlyLifts.csv "Heavenly Lifts" "{'color': 'black',  'weight':2, 'opacity':0.75, 'dash_array': '14 4'}" "Heavenly Mountain Resort" \
		-skirun /home/don/GPX/SkiAreas/HeavenlyRuns.csv "Heavenly Runs" "{'color': 'olive',  'weight':2, 'opacity':0.5}" "Heavenly Mountain Resort" \
		-photos /home/don/Pictures/tahoe_ski_2022_q4 /dvo/images "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-profile \
		-debug 1
	sed -e 's/"dashArray": "class_/"className": "/' < test99_x.html > test99.html

#		-skitracks /home/don/GPX/Tracks/SkiingTahoe_22-23/20221216_0937-20221216_1510_GPX.gpx "DVO Ski tracks" "{'color': 'pink',  'weight':2, 'opacity':0.5}" "Heavenly Mountain Resort" \
#		-skitracks /home/don/GPX/Tracks/SkiingTahoe2022/20221122_Heavenly.gpx "DVO Runs at Heavenly" \



all: DvoTahoe2022.html

DvoTahoe2022.html:
	time python3 gps_pix_map.py \
		-outfile dvoTahoe2022.html \
		-tracks  /home/don/GPX/Tracks/TahoeRimTrail2022  https://kokanee.mynetgear.com/dvo/tracks "Don's Hiking Tahoe Rim Trail (orange)" "{'color': 'orange', 'weight': 4, 'opacity': 0.5}" \
		-tracks  /home/don/GPX/Tracks/HikingTahoe2022    https://kokanee.mynetgear.com/dvo/tracks "Don's Hiking Tahoe (brown)"            "{'color': 'brown',  'weight': 4, 'opacity': 0.5}" \
		-tracks  /home/don/GPX/Tracks/HikingKokanee2022  https://kokanee.mynetgear.com/dvo/tracks "Don's Hiking from Kokanee (red)"       "{'color': 'red',    'weight': 4, 'opacity': 0.5}" \
		-tracks  /home/don/GPX/Tracks/KayakTahoe2022     https://kokanee.mynetgear.com/dvo/tracks "Don's Kayaking (blue)"                 "{'color': 'blue',   'weight': 4, 'opacity': 0.5}" \
		-skitracks /home/don/GPX/Tracks/SkiingTahoe_22-23/20221216_Heavenly.gpx "DVO Ski tracks (black)" "{'color': 'pink',  'weight':2, 'opacity':0.5}" \
		-photos  /home/don/Pictures/tahoe_hike_2022_q1 https://kokanee.mynetgear.com/dvo/hkgpix                       "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-photos  /home/don/Pictures/tahoe_hike_2022_q2 https://kokanee.mynetgear.com/dvo/hkgpix                       "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-photos  /home/don/Pictures/tahoe_hike_2022_q3 https://kokanee.mynetgear.com/dvo/hkgpix                       "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-photos  /home/don/Pictures/tahoe_hike_2022_q4 https://kokanee.mynetgear.com/dvo/hkgpix                       "{'color': 'purple', 'weight': 0.5, 'fill_color': 'lightblue', 'radius': 6}" \
		-fixed   /home/don/GPX/Routes/trt-500pointsPerTrack.gpx "Tahoe Rim Trail (yellow)"      "{'color':'yellow',  'weight': 2}" '<a href="http://www.tahoerimtrail.org">Tahoe Rim Trail</a>' \
		-pm_group 200 \
		-polygon /home/don/GPX/SkiAreas/Heavenly.gpx     "Heavenly Mountain Resort (approx bdry)" "{'color': 'purple', 'dash_array': '16 8'}" '<a href="http://www.skiheavenly.com">Heavenly Mountain Resort<\a>' \
		-skilifts   /home/don/GPX/SkiAreas/HeavenlyLifts.csv "Heavenly Lifts"                      "{'color': 'black',  'weight':2, 'opacity':0.75, 'dash_array': '14 4'}" "Heavenly" \
		-polygon /home/don/GPX/SkiAreas/Kirkwood.gpx "Kirkwood Mountain Resort (approx bdry)" "{'color': 'grey', 'dash_array': '16 8'}" '<a href="http://www.kirkwood.com">Kirkwood<\a>' \
		-skilifts   /home/don/GPX/SkiAreas/KirkwoodLifts.csv    "Kirkwood Lifts"                 "{'color': 'grey',  'weight':2, 'opacity':0.75, 'dash_array': '14 4'}" "Kirkwood" \
		-showbbox 2 \
		-cachecorrections ~/.gps_pix_map/jpeg_gps_corrections.csv \
		-include 39 -120  50 \
		-exclude 38.92828 -120.02159 38.93360 -120.00747 \
		-exclude 38.92928 -120.01763 38.93602 -120.00478 \
		-profile \
		-css_url leaflet_overloaded_css https://kokanee.mynetgear.com/dvo/hiking_map.css \
		-slideshow ./slideshow.html \
		-ss_url_prefix https://kokanee.mynetgear.com/dvo/images \
		-ss_pace 5000 \
		-checkbox_how day \
		-debug 1  2>&1  | tee output_7q.txt
	sed -e 's/"dashArray": "class_/"className": "/' < dvoTahoe2022.html > DvoTahoe2022.html

# gps_pix_map

---

## An executable Python script to create a HTML page with your GPS tracks and photos' locations.

A python script intended to be run from the command line and controlled with various command-line arguments.

Reads GPX files as input, combined with the GPS-aware EXIF from image files to produce a **.html** file which can be served on a website or displayed locally with most browsers.

Requires folium, gpxpy and numerous more common Python modules.

[![gps_pix_map_demo](http://kokanee.mynetgear.com/demo_photos/gps_pix_map_demo_scaled.png)](http://kokanee.mynetgear.com/gps_pix_map_demo.html "Link to a live HTML page generated by the command-lines below.")

## Usage

1. Start with GPX and/or images with EXIF from one or more excursions.

    * An excursion is some traveling (walking, running, hiking, biking, skiing, kayaking, driving, riding, flying, etc.) that is being tracked with a GPS device. The GPS device may be a cell-phone with an app that can output in the GPX format.
    * EXIF is a collection of data that many cameras and and most cell-phones embed in the file containing an image. EXIF supports geo-tagging with GPS data. gps_pix_map.py relies on finding that information in the image files.
2. Upload these files - the GPS tracking in GPX format, and image files - to the computer on which you will run this Python script.
3. Determine the suitable command-line arguments to identify those files, and run this gps_pix_map.py script.

    * The script produces a HTML file that can be rendered with most browsers.

```
python3 gps_pix_map.py \
        -outfile gps_pix_map_demo1.html \
        -tracks ./hiking_tracks "Some hikes (red)" "'color': 'red'" \
        -tracks ./kayaking_tracks "Kayaking {blue)" "'color': 'blue'" \
        -photos ./demo_photos ./demo_photos "'color': 'purple', 'radius': 6"

sed -e 's/"dashArray": "class_/"className": "/' < gps_pix_map_demo1.html > gps_pix_map_demo.html
```

4. Optionally, copy the HTML and the image files to your web-site.
5. Optionally, to view using the firefox browser:

    `firefox -new-tab gps_pix_map_demo.html`

In the above example, the directories **./hiking_tracks** and **./kayaking_tracks** each contain some **.gpx** files. All of those files will be rendered as tracks on the map shown in the **.html** file.
Additionally, image files containing EXIF data with GPS information is in the **-photos** directory.
The additional information in the **-tracks** and **-photos** command-line arguments provide labeling and other information.

**Note**: The **sed** command above is part of a work-around to a possible bug in the underlying **folium** python package. I'm still investigating.

## Help

If the command-line arguments include **-h**, then a help message is printed and the script terminates.

    python3 gps_pix_map.py -h

import sys
import json
from pathlib import Path

import math
import os
import argparse
import folium
from folium import plugins
import gpxpy
import matplotlib
import numpy as np
import pandas as pd
from exif import Image as ExifImage
import webbrowser

import time
import datetime
from datetime import timezone

from PIL import Image as PilImage


def StringToDict(the_string):
    """the_string should resemble "{key1: value1, key2: value2}" or "key1: value1, key2: value2"
    Allows the user to pass in, via command line arguments, arbitrary arguments to be passed
    to underlying folium calls. Thus it is up to the user to determine the correct names of the keys.
    As an example, here are some keys that work ok with the folium Polylines() call:
        "color", "weight", "dash_array"
        although be aware of the post-processing hack on dash_array=class_ - see comments elsewhere
    """
    if (the_string is None) or (len(the_string) == 0): return {}
    if the_string[0] != '{': the_string = '{' + the_string
    if the_string[-1] != '}': the_string = the_string + '}'
    try:
        # https://www.geeksforgeeks.org/python-convert-string-dictionary-to-dictionary/
        the_dict = eval(the_string)
        return the_dict
    except BaseException as e:
        print("Exception while attempting to eval({}): {}, {}".format(the_string, type(e), e))
    return {}


def UpdateDataFrame(original_df, updater_df):
    """
    Returns a new DataFrame that is a variation of original_df, such that any new or different information from
    updater_df is used instead. So if both DFs have a cell at the same row/index and column, but the values differ, then
    the value from updater_df is used. Any new rows or new columns from updater will also be added to the original_df.
    """
    # I tried many variations of doing this directly with the dataframes - using merge(), merge_ordered(), update(),
    # join(), concat() but couldn't find any way to get what I wanted (which seems relatively simple).

    dict1 = original_df.to_dict(orient="index")
    dict2 = updater_df.to_dict(orient="index")
    for key in dict1:
        if key in dict2:
            dict1[key].update(dict2[key])
    for key in dict2:
        if not key in dict1:
            dict1[key] = dict2[key]
    new_df = pd.DataFrame.from_dict(dict1, orient="index")
    new_df2 = new_df.where(pd.notnull(new_df), None)

    return new_df2


# Regarding times and dates...
# I had much more problems dealing with times and dates than I expected. Here's a summary of some of the difficulties
# and where things are now...
# 1) I needed to compare times/dates from different sources (the GPS device and the EXIF in images), therefore all my
#       times/dates needed to be "timezone aware" (rather than "timezone naive" - google this if not familiar).
# 2) The times/dates returned from my GPS device via gpxpy were in UTC but without the timezone set (and therefore
#       timezone-naive).
# 3) To allow comparisons, python seemed to required that times/dates be in the same timezone (which seems overly
#       restrictive). So I intend to have all times/dates internally in UTC and convert to/from as needed (such as for
#       display - in which case the local timezone is generally preferred).
# 4) I still don't think I have it figured out how to deal appropriately with the local timezone - i.e. to determine
#       what it is in a robust manner.
# 5) Getting the date/time from EXIF of when an image was taken was more difficult than expected since various fields
#       might or might not be present; see get_exposure_datetime_from_EXIF() - still a work-in-progress.

class TzHelper:
    # https://stackoverflow.com/questions/4563272/how-to-convert-a-utc-datetime-to-a-local-datetime-using-only-standard-library/13287083#13287083
    local_tz = timezone(datetime.timedelta(seconds=-time.timezone))

    def insert_timezone(
            datetime_arg):  # This shifts the timezone without changing the date/tima values to compensate (which
                            # effectively changes the time if converted to UTC)
        # So use this on naive date/times we know to be in local timezone.
        converted_datetime = datetime_arg.replace(tzinfo=TzHelper.local_tz)
        return converted_datetime

    def as_local(datetime_arg : datetime):
        return datetime_arg.astimezone(TzHelper.local_tz)


def get_time() -> float:
    """Returns the number of seconds (with milliseconds or better resolution) since the "epoch" (look it up). Primary
    use is to take the difference between two calls to determine the time interval (in seconds) - to help analyze
    run-time performance, etc.
    """
    return time.time()


time_program_start = get_time()

checkbox_datestamps = set()  # Collection of all checkbox_datestamps used in photos and/or hiking tracks. Should be in local-time (not
                    # UTC). This is used to create the menu checkbox control, so whatever resolution (year, month-year,
                    # day-month-year, etc.) that is wanted for that user-control, should be used here.


EARTH_CIRCUMFERENCE_FEET = 24902 * 5280
EARTH_FEET_PER_DEGREE = EARTH_CIRCUMFERENCE_FEET / 360


def degrees_to_feet(degrees: float) -> float:
    return degrees * EARTH_FEET_PER_DEGREE


def feet_to_degrees(feet: float) -> float:
    return feet / EARTH_FEET_PER_DEGREE


def None_to_nan(value):
    return value if (value != None) else np.nan


debug_level = 0  # 0 is off, 1 is basic higher numbers show more, 10 is verbose

track_color = "red"
track_weight = 3.5
track_opacity = 0.4

Overall_min_latitude = None
Overall_max_latitude = None
Overall_min_longitude = None
Overall_max_longitude = None


def OverallBBoxCheck(latitude: float, longitude: float):
    global Overall_min_latitude, Overall_max_latitude, Overall_min_longitude, Overall_max_longitude
    if (Overall_min_latitude is None) or (latitude < Overall_min_latitude):  Overall_min_latitude = latitude
    if (Overall_max_latitude is None) or (latitude > Overall_max_latitude):  Overall_max_latitude = latitude
    if (Overall_min_longitude is None) or (longitude < Overall_min_longitude): Overall_min_longitude = longitude
    if (Overall_max_longitude is None) or (longitude > Overall_max_longitude): Overall_max_longitude = longitude


def CreateBBox(inlist):
    if len(inlist) == 4:
        return inlist
    if len(inlist) == 3:  # lat,lon of center point, plus edge length
        center_lat = inlist[0]
        center_lon = inlist[1]
        edge_length_miles = inlist[2]
        half_edge_degrees_lat = feet_to_degrees(edge_length_miles * 5280) / 2
        half_edge_degrees_lon = half_edge_degrees_lat / math.cos(math.radians(center_lat))
        sw = (center_lat - half_edge_degrees_lat, center_lon - half_edge_degrees_lon)
        ne = (center_lat + half_edge_degrees_lat, center_lon + half_edge_degrees_lon)
        return (sw[0], sw[1], ne[0], ne[1])
    # Else - an error
    print("ERROR in CreateBBox({}) - expecting either 3 or 4 values.".format(inlist))


def in_coordinate_box(point: tuple, box: tuple) -> bool:
    """Determine if a GPS point is within a bounding box.
    Args:
        point is latitude and longitude as three floating point values (latitude, longitude, and altitude)
        box is six floating point values - such as (lat-min, lon-min, alt-min, lat-max, lon-max, alt-max)
    Returns:
        bool: True if the point is in the box (or exactly on the boundary). False otherwise.
    """
    return ((point[0] >= min(box[0], box[3])) and (point[0] <= max(box[0], box[3])) and  # latitude
            (point[1] >= min(box[1], box[4])) and (point[1] <= max(box[1], box[4])) and  # longitude
            (point[2] >= min(box[2], box[5])) and (point[2] <= max(box[2], box[5])))  # altitude


def accessible_point(point: tuple, include_boxes=None, exclude_boxes=None, show_rejections: bool = False) -> bool:
    if (exclude_boxes != None) and (len(exclude_boxes) > 0):
        for the_box in exclude_boxes:
            if (point[0] >= the_box[0]) and (point[0] <= the_box[2]) and (point[1] >= the_box[1]) and (
                    point[1] <= the_box[3]):  # point inside of this exclude box?
                if show_rejections: print(
                    "********   Rejecting point ({:.6f},{:.6f})  Exclude box: ({:.6f},{:.6f})   ({:.6f},{:.6f})".format(
                        point[0], point[1], the_box[0], the_box[1], the_box[2], the_box[3]))
                return False
    if (include_boxes != None) and (len(include_boxes) > 0):
        for the_box in include_boxes:
            if (point[0] >= the_box[0]) and (point[0] <= the_box[2]) and (point[1] >= the_box[1]) and (
                    point[1] <= the_box[3]):  # point inside of this include box?
                return True
        if show_rejections: print(
            "********   Rejecting point ({:.6f},{:.6f})  Not in any include box: {}".format(point[0], point[1],
                                                                                            include_boxes))
        return False
    else:
        return True  # There isn't an include list - so include everything


def GPS_DMS_to_degrees(coord: tuple, ref: str) -> float:
    """Covert a tuple of a coord (i.e. either latitude or longitude) in the format (degrees, minutes, seconds) and a
    directional reference - as generally extracted from EXIF - to a floating representation.
    Args:
        coord (tuple[float,...]): A tuple of degrees, minutes and seconds - such as in EXIF GPSLatitude
        ref: One of "N", "S", "E" or "W". - such as in GPSLatitudeRef (or GPSLongitudeRef)
    Returns:
        float: A signed float of decimal representation of the coordinate.
    """
    if (coord is None) or (coord[0] is None) or (coord[1] is None) or (coord[2] is None) or (ref is None):
        return None

    float_value = coord[0] + coord[1] / 60 + coord[2] / 3600
    if ref.upper() in ['W', 'S']:   return float_value * -1
    elif ref.upper() in ['E', 'N']: return float_value
    else:
        print("WARNING: Unexpected Incorrect coordinate reference={}. Expecting one of 'N', 'S', 'E' or 'W'.".format(ref))
        return None


def get_exposure_datetime_from_EXIF(exif_dict, filename=None):
    """Intended to be a robust routine to extract the date/time (timezone aware) of an image's exposure based on
    the EXIF fields. The challenge is that not all images's have the same EXIF fields. So this is basically a sequence
    of try-this and then try-that. The exif_dict is a python dictionary - key is an EXIF attribute. Which attributes
    are used depends on the approach.
    see also: https://www.cipa.jp/std/documents/e/DC-X008-Translation-2019-E.pdf
    """
    the_datetime = None
    # First try to get a datetime (without a timezone yet)
    for exif_attribute in ["datetime_original", "datetime", "datetime_digitized"]:
        if (exif_attribute in exif_dict) and (exif_dict[exif_attribute] != None):
            the_datetime = datetime.datetime.strptime(exif_dict[exif_attribute], "%Y:%m:%d %H:%M:%S")
            break  # Partial success! (so far - doesn't yet have a timezone)
    if the_datetime != None:  # Now try to determine timezone
        utc_offset = the_datetime.strftime("%z")  # Will be empty if timezone naive (which is different than 0 or UTC)
        if len(utc_offset) != 0:  # Assume this indicates a valid timezone
            return the_datetime  # Success (variation 1)
        else:  # Need to determine the timezone
            for exif_attribute in ["offset_time_original", "offset_time",
                                   "offset_time_digitized"]:  # Should be in form +HH:MM (or -HH:MM)
                if (exif_attribute in exif_dict) and (exif_dict[exif_attribute] != None):
                    after_split = exif_dict[exif_attribute].split(":")
                    the_timezone = datetime.timezone(
                        datetime.timedelta(hours=int(after_split[0]), minutes=int(after_split[1])))
                    the_datetime = datetime.datetime(the_datetime.year, the_datetime.month, the_datetime.day,
                                                     the_datetime.hour, the_datetime.minute, the_datetime.second,
                                                     tzinfo=the_timezone)
                    return the_datetime  # Success (variation 2)

    # Get here if haven't found a valid timezone (and/or maybe not a valid datetime). Alternative is try to decode the
    # gps_datestamp and gps_timestamp - assuming that is at (about) the same time as when the picture was taken - in
    # the UTC timezone. Formats should be: gps_datestamp: YYYY:MM:DD
    #                   gps_timestamp a tuple consisting of hour, minutes and seconds
    if (    ("gps_datestamp" in exif_dict) and (exif_dict["gps_datestamp"] is not None)
        and ("gps_timestamp" in exif_dict) and (exif_dict["gps_timestamp"] is not None)):
        gps_time_stamp = exif_dict["gps_timestamp"]
        gps_datetime = (datetime.datetime.strptime(exif_dict["gps_datestamp"], "%Y:%m:%d").replace(tzinfo=datetime.timezone.utc)
                        + datetime.timedelta(hours=gps_time_stamp[0], minutes=gps_time_stamp[1], seconds=gps_time_stamp[2]))
        return gps_datetime  # Success (variation 3)

    if the_datetime is not None:  # This ia a timezone naive - how to add/set for the local timezone?
        if True:
            # https://stackoverflow.com/questions/4563272/how-to-convert-a-utc-datetime-to-a-local-datetime-using-only-standard-library/13287083#13287083
            local_tz = timezone(datetime.timedelta(seconds=-time.timezone))
            the_datetime = the_datetime.replace(tzinfo=local_tz)
            return the_datetime  # succeess? (variation 4)
        else:
            return TzHelper.insert_timezone(the_datetime)

    try:
        basename = os.path.basename(filename)
        filename_datetime = datetime.datetime.strptime(basename,
                                                       "%Y%m%d_%H%M%S.jpg")  # Specific filename format for my Samsung/Android phone - uses creating date/time. Are there others to support?
    except ValueError as e:
        print("Caught VaueError exception for strptime({}, %Y%m%d_%H%M%S.jpg: {}".format(basename, e))
        # Not generating an error here - just move on to the next approach
        filename_datetime = None
    if filename_datetime != None:
        return filename_datetime  # Succss (variation 5)

    # Perhaps try using the creation date of the file itself - but I'm concerned on Linux systems it would be the date
    # the file was copied to the computer from the camera
    print("WARNING - can't determine time and date for {}".format(filename))
    return None  # Failure


def get_EXIF_from_directory(directory: Path, href_str: str, file_suffix: str = '*.jpg', after: datetime=None,
                            profile: bool = False, debug: bool = False) -> pd.DataFrame:
    """Create a dictionary of spatial data from the EXIF of photos in a folder.
    Args:
        directory: Where to search for files (not recursively)
        href: Use this in generating the <a> HTML tag.
        file_suffix: extension of images to read.
        after - optional - as a time optimization, don't look at any files before this datetime
        profile: generate some run-time timing information if enabled
        debug: more run-time logging if enabled

    Returns:
        DataFrame - with the filename as the index and a column for each for the EXIF fields (below) - so one row
            per matching file in the folder.
    """
    required_exif_fields = ["gps_latitude", "gps_latitude_ref", "gps_longitude", "gps_longitude_ref", "gps_altitude"]
    exif_fields = required_exif_fields + ["datetime", "datetime_original", "offset_time", "offset_time_original",
                                          "offset_time_digitized",
                                          "gps_timestamp", "gps_datestamp",
                                          "focal_length", "make", "model", "digital_zoom_ratio", "image_width",
                                          "image_height", "orientation"]
    non_exif_fields = ["filename", "latitude", "longitude", "timestamp", "href", "comment"] # will be DF column headers
    time_func_start = get_time()

    coord_df = pd.DataFrame(columns=non_exif_fields + exif_fields)
    if after is None:
        source_files = [f for f in Path(directory).rglob(file_suffix)] if (directory is not None) else []
    else:
        after_ts = datetime.datetime.timestamp(after)
        source_files = [f for f in Path(directory).rglob(file_suffix) if (f.stat().st_mtime >= after_ts)] if (
                directory is not None) else []
    time_func_find_files = get_time()
    if profile:
        print("Expanded {} sourcefiles: {:.6f}".format(len(source_files), time_func_find_files - time_func_start))

    file_count = 0
    added_file_count = 0
    for f in source_files:
        str_f = str(f)
        file_count += 1
        if debug:
            print("In read_geotagged_photos_from_dir(folder={},...): file #{}: {}".format(directory, file_count, f))
        exif_dict = read_exif_data(f, exif_fields, debug=debug)
        latitude  = GPS_DMS_to_degrees(exif_dict.get("gps_latitude", None), exif_dict.get("gps_latitude_ref", None))
        longitude = GPS_DMS_to_degrees(exif_dict.get("gps_longitude", None), exif_dict.get("gps_longitude_ref", None))
        altitude = exif_dict.get("gps_altitude", None)
        new_row_dict = exif_dict
        new_row_dict["filename"] = str_f
        new_row_dict["latitude"] = latitude
        new_row_dict["longitude"] = longitude
        new_row_dict["href"] = href_str
        new_row_dict["comment"] = None
        new_timestamp = get_exposure_datetime_from_EXIF(exif_dict, str_f)
        if new_timestamp is not None:
            utc_timestamp = new_timestamp.astimezone(timezone.utc)
            new_row_dict["timestamp"] = utc_timestamp
        # coord_df = coord_df.append(new_row_dict, ignore_index=True) # Deprecated due to run-time FutureWarning
        coord_df = pd.concat([coord_df, pd.DataFrame.from_records([
            new_row_dict])], sort=False)  # https://stackoverflow.com/questions/70837397/good-alternative-to-pandas-append-method-now-that-it-is-being-deprecated
        added_file_count += 1

    time_after_loop = get_time()
    if profile: print("Time to loop thru {} {} files in folder {}: {:.6f} ({} files skipped or excluded)".format(
        file_count, file_suffix, directory, time_after_loop - time_func_find_files, file_count - added_file_count))
    return coord_df


def rgba_to_hex(rgba: tuple):
    return ('#{:02X}{:02X}{:02X}').format(*rgba[:3])


def read_exif_data(file_path: Path, requested_exif_attributes: [], debug: bool = False) -> {}:
    """Find the indicated EXIF attributes from the photo. Return the values in a list."""
    output_dict = {}
    with open(file_path, 'rb') as f:
        the_image = ExifImage(f)
        if debug:
            print("In read_exif_data(file={},requested_exif_attributes={},...)".format(file_path,
                                                                                       requested_exif_attributes))
            print("has_exif={}".format(the_image.has_exif))
            if the_image.has_exif:
                print("ExifImage({})".format(the_image))
                for zz in the_image.list_all():
                    print("    {:30s}:".format(zz), end='', flush=True)
                    if zz == "user_comment":
                        print(" skipping")
                    else:
                        try:
                            print(" {}".format(the_image.get(zz, default="(none)")))
                        except BaseException as e:
                            print("Exception {}: {}".format(type(e), e))

        if the_image.has_exif:
            for exif_attribute in requested_exif_attributes:
                value = the_image.get(exif_attribute, default=None)
                output_dict[exif_attribute] = value
    return output_dict


def create_thumbnail_copy_if_needed(photo_filename, thumbnail_directory, suffix="_thumb", max_size=(200, 200)) -> str:
    """Create a thumbnail copy from the photo_filename argument - and store in the thumbnail_directory.
    Do nothing if the thumbnail already exists (doesn't check size or integrity).
    """
    # Create the modified photo_filename in the thumbnail_directory and containing suffix (if any)
    (dir_part, file_part) = os.path.split(photo_filename)
    (root_part, ext_part) = os.path.splitext(file_part)
    thumbnail_name = thumbnail_directory + "/" + root_part + suffix + ext_part
    # See if the file already exists
    if os.path.exists(thumbnail_name):
        return thumbnail_name
    # print("photo={}, thumbdir={}, new_name={}".format(photo_filename, thumbnail_directory, thumbnail_name))

    # Now create the thumbnail file
    original_image = PilImage.open(photo_filename)
    copied_original_image = original_image.copy()
    copied_original_image.thumbnail(max_size)
    copied_original_image.save(thumbnail_name)
    return thumbnail_name


def rows_with_identical_values_for_indicated_columns2(df: pd.DataFrame, column_headers: []) -> []:
    """
    """
    column0_1st = df.iloc[0][column_headers[0]]
    column1_1st = df.iloc[0][column_headers[1]]
    new_df = df.loc[(df[column_headers[0]] == column0_1st) & (df[column_headers[1]] == column1_1st)]
    return new_df


def read_gpx_file(filename: Path, include_datetime=False, profile: bool = False):  # returns a list of tracks
    t1 = get_time()
    if profile: print("File:({})".format(filename), end='', flush=True)

    # Get EXIF info
    with open(filename, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    # Make a list of list of tuples. Each tuple is a point. The list of points is a path (for one track). The list of
    # paths (covering all tracks) is what gets returned.
    # Question - should each segment be a separate line? If so, need to add another level of nesting of the lists
    # (possible future feature? - I don't have any test data).
    first_time = None
    list_of_tracks = []
    for track in gpx.tracks:
        points = []  # list of tuples - for one path (one track)
        for segment in track.segments:
            for point in segment.points:
                if (include_datetime or first_time is None) and (point.time is not None):
                    utc_time = point.time.replace(tzinfo=datetime.timezone.utc) # As above - no timezone in gpxpy
                    if first_time is None:
                        first_time = utc_time
                if include_datetime:
                    points.append(tuple([point.latitude, point.longitude, utc_time]))
                else:
                    points.append(tuple([point.latitude, point.longitude]))
        list_of_tracks.append((track.name, points, segment))

    if profile: print(" - took {:.6f} seconds".format(get_time() - t1))
    return list_of_tracks, first_time


def attempt_to_recover_GPS_coords_in_images(image_df, gpx_files, debug_level=0, show_changes=False, profile=False):
    if debug_level >= 2: print("Start of attempt_to_recover_GPS_coords_in_images((...,{},{})".format(gpx_files, debug_level))
    image_df.loc[:, 'original_lat'] = image_df.loc[:, 'latitude']
    image_df.loc[:, 'original_lon'] = image_df.loc[:, 'longitude']
    corrected_count = 0
    for thing in gpx_files:
        for filename in sorted(os.listdir(thing), reverse=True):
            if filename.endswith(".gpx"):
                list_of_tracks, first_time_not_used = read_gpx_file(thing + "/" + filename, include_datetime=True, profile=profile)
                if debug_level >= 10:
                    print("list_of_tracks=", list_of_tracks)
                for track_pair in list_of_tracks:
                    first_time = pd.to_datetime(track_pair[1][0][2], utc=True)
                    last_time = pd.to_datetime(track_pair[1][-1][2], utc=True)
                    image_df_selected_rows = image_df.loc[
                        (image_df['timestamp'] >= first_time) & (image_df['timestamp'] <= last_time)]

                    if len(image_df_selected_rows) > 0:  # This means the image(s) in image_df_selected_rows were snapped sometime during the track (in track_pair).
                        # Now find the points in that track that are closest in time to when the image was snapped.
                        # I think there is a more pythonic way to do this
                        for image_filename, row in image_df_selected_rows.iterrows():
                            last_before_time = None
                            last_before_point = ()
                            first_after_time = None
                            first_after_point = ()
                            count = 0
                            for lat, lon, timestamp in track_pair[1]:
                                count += 1
                                if timestamp <= row['timestamp']:
                                    last_before_time = timestamp
                                    last_before_lat = lat
                                    last_before_lon = lon
                                if timestamp >= row[
                                    'timestamp']:  # Not else if - want both values the same if an exact time match (however unlikely)
                                    first_after_time = timestamp
                                    first_after_lat = lat
                                    first_after_lon = lon
                                    break
                            if debug_level >= 3: print(
                                "In attempt_to_recover_GPS_coords_in_images():  Between ({:6f},{:.6f}) and ({:.6f},{:.6f})".format(
                                    last_before_lat, last_before_lon, first_after_lat, first_after_lon))
                            if debug_level >= 3: print(
                                "In attempt_to_recover_GPS_coords_in_images():  Time difference before: {}, Time difference after: {}".format(
                                    row['timestamp'] - last_before_time, first_after_time - row['timestamp']))
                            ratio = 0
                            if last_before_time == row['timestamp']:
                                ratio = 0
                            elif first_after_time == row['timestamp']:
                                ratio = 1
                            else:
                                ratio = (row['timestamp'].astimezone(
                                    datetime.timezone.utc) - last_before_time.astimezone(datetime.timezone.utc)) / \
                                        (first_after_time.astimezone(
                                            datetime.timezone.utc) - last_before_time.astimezone(datetime.timezone.utc))
                            # Interpolate
                            new_lat = ratio * (first_after_lat - last_before_lat) + last_before_lat
                            new_lon = ratio * (first_after_lon - last_before_lon) + last_before_lon
                            if show_changes:
                                distance_meters = gpxpy.geo.distance(row['latitude'], row['longitude'], None, new_lat,
                                                                     new_lon, None)
                                if distance_meters >= 300:  # Arbitrary switch-over point - want to show longer distances in miles, shorter distances in feet
                                    distance = "{} miles".format(round(distance_meters / 1609, 1))
                                else:
                                    distance = "{:.0f} feet".format(round(distance_meters * 3.28084, 0))

                                print(
                                    "GPS coordinate correction for {} from ({:.6f},{:.6f}) to ({:.6f},{:.6f}): Distance={}".format(
                                        image_filename, row['latitude'], row['longitude'], new_lat, new_lon,
                                        distance, ))
                            # Any of the following 3 lines will produce the dreaded SettingWithCopyWarning - even though I'm doing what (I think) the message recommends.
                            image_df.loc[
                                image_filename, 'comment'] = "GPS location corrected from {:.6f},{:.6f}".format(
                                row['latitude'], row['longitude'])
                            image_df.loc[image_filename, 'latitude'] = new_lat
                            image_df.loc[image_filename, 'longitude'] = new_lon
                            corrected_count += 1
    image_df["comment"] = np.where(
        (image_df["latitude"] == image_df["original_lat"]) & (image_df["longitude"] == image_df["original_lon"]),
        "Location is suspect, but couldn't correct via data in the tracks.", image_df["comment"])
    return image_df, corrected_count


class XYZ:
    date_to_feature_map = {}  # class variable (ala C++'s static?)

    @classmethod
    def get_datetime_featuregroup(cls, datetime_arg):
        # Covert datetime to desired string - i.e. with only year, or year/month or year/month/day - as indicated by options
        date_string = ""
        if False:  # Convert to year (only)
            date_string = datetime_arg.strftime("%Y")
        elif True:  # Convert to year and month
            date_string = datetime_arg.strftime("%Y-%b")
        elif False:  # Convet to year and week number (starting on Sunday)
            date_string = datetime_arg.strftime("%Y-week%U")
        else:  # Specific date (year, month, day)
            date_string = datetime_arg.strftime("%Y-%b-%-d")

        if not (date_string in cls.date_to_feature_map):
            cls.date_to_feature_map["date_string"] = folium.FeatureGroup(date_string)
        return cls.date_to_feature_map["date_string"]


def remove_excluded_points(list_of_points, exclude_boxes, show_rejections: bool = False):
    return_list = []
    for point in list_of_points:
        if accessible_point(point, None, exclude_boxes, show_rejections=show_rejections):
            return_list.append(point)
    return return_list


class ImagePopupHelper:
    def __init__(self, df_len):
        self.df_len = df_len
        self.lat_sum = 0
        self.lon_sum = 0
        self.popup_str = '<table class="image-list-popup">' if self.df_len > 1 else ""
        self.row_start = "<tr><td>" if self.df_len > 1 else ""
        self.row_mid = "</td><td>" if self.df_len > 1 else "&nbsp"
        self.row_end = "</td></tr>\n" if self.df_len > 1 else ""
        self.timestamps = set()

    def photo_marker_add_row(self, lat_round, lon_round, timestamp, image_width, image_height, orientation, filename, make,
                             model, digital_zoom_ratio, href, comment):
        if lat_round is not None: self.lat_sum += lat_round
        if lon_round is not None: self.lon_sum += lon_round

        if timestamp is not None:
            self.timestamps.add(timestamp)

        base_filename = os.path.basename(filename) if filename is not None else ""
        lens_type = ""
        if make in ["samsung"]:
            if model in ["SM-A536U1", "SM-G955U"]:
                if digital_zoom_ratio is not None:
                    lens_type = "panorama" if image_width > (2.5 * image_height) else \
                        "super wide" if (digital_zoom_ratio == 0) else \
                            "wide" if (digital_zoom_ratio == 1.0) else \
                                "normal" if (digital_zoom_ratio <= 2.0) else \
                                    "telephoto" if (digital_zoom_ratio <= 4.0) else \
                                        "long telephoto"

        # width and height might need to be reversed based on orientation: http://sylvana.net/jpegcrop/exif_orientation.html
        oriented_width = image_width if orientation < 4 else image_height
        oriented_height = image_height if orientation < 4 else image_width

        self.popup_str = self.popup_str + self.row_start + \
                         '<a href="{}/{}" target="_blank" rel="noopener noreferrer">{}</a>{}{}{}{}x{}{}{}{}{}{}{}'.format(
                             href,
                             base_filename, base_filename, self.row_mid,
                             lens_type, self.row_mid,
                             oriented_width, oriented_height, self.row_mid,
                             TzHelper.as_local(timestamp).strftime("%-d-%b-%Y"), self.row_mid,
                             TzHelper.as_local(timestamp).strftime("%-I:%M %p"), self.row_mid,
                             comment if comment is not None else ""
                         ) + self.row_end


def create_photo_marker(df, include_boxes, exclude_boxes, show_rejections, the_photos_args_dict):
    df_len = len(df)
    df.sort_index(inplace=True)

    popup_gen = ImagePopupHelper(df_len)
    [popup_gen.photo_marker_add_row(latitude, longitude, timestamp, image_width, image_height, orientation, filename, make,
                                    model, digital_zoom_ratio, href, comment) \
     for
     latitude, longitude, timestamp, image_width, image_height, orientation, filename, make, model, digital_zoom_ratio, href, comment
     in zip(
        df['latitude'], df['longitude'], df['timestamp'], df['image_width'], df['image_height'], df['orientation'],
        df['filename'], df['make'], df['model'], df['digital_zoom_ratio'], df['href'], df['comment'])]

    if df_len > 1:
        popup_gen.popup_str = popup_gen.popup_str + "</table>"

    ts_string = ""
    for ts in sorted(popup_gen.timestamps):
        formatted_timestamp = TzHelper.as_local(ts).strftime("%Y%m%d")
        checkbox_datestamps.add(formatted_timestamp)
        ts_string += formatted_timestamp + " "
    lat_mean = popup_gen.lat_sum / df_len
    lon_mean = popup_gen.lon_sum / df_len
    if accessible_point((lat_mean, lon_mean), include_boxes, exclude_boxes, show_rejections):
        the_marker = folium.CircleMarker(
            [lat_mean, lon_mean]
            , **the_photos_args_dict
            , tooltip="{} image{}".format(df_len, "s" if df_len > 1 else "")
            , popup=popup_gen.popup_str
            # Tried lots of variations - but couldn't get the className stuff to pass information into the generated .html file
            # ,style={"className" :"20220802", "class_name": "20220802", "class" : "20220802", "class_list" : "20220802"}
            # ,kwargs={"className" :"20220802", "class_name": "20220802", "class" : "20220802", "class_list" : "20220802"}
            # ,class ="20220802"
            # ,className ="20220802"
            # ,class_name ="20220802"
            # ,class_list ="20220802"
            , dash_array="class_" + formatted_timestamp
            # This is a kluge - apparently folium doesn't support class_name="..." (documentation
            # implies it should), so I'm picking a little used attribute (dashArray) that is supported and hijacking it.
            # Relying on a post-processing step in the generated .html file to change the generated "dashArray" to the desired "className"
            # There's another near-identical kluge elsewhere in this file.
        )
        OverallBBoxCheck(lat_mean, lon_mean)
        the_marker.add_to(photo_group)
        photo_group.add_to(folium_map)
    return


def deal_with_cache_corrections_file(jpeg_df, filename):
    """Read the CSV from filename (if any) and merge with the jpeg_df argument. Then save the merged result back
    to the CSV.
    """
    if os.path.isfile(filename):
        try:
            original_csv = pd.read_csv(filename, index_col="filename")
        except IOError as e:
            print("WARNING: error while attempting to read {}: {}".format(filename, e))
            original_csv = pd.DataFrame()
    else:
        original_csv = pd.DataFrame()

    jpeg_df = UpdateDataFrame(jpeg_df, original_csv)

    try:
        jpeg_df.loc[:, ["latitude", "longitude", "original_lat", "original_lon", "comment"]].sort_index().to_csv(
            filename, header=True, index=True, index_label="filename")
    except IOError as e:
        print("WARNING: error while attempting to save {}: {}".format(filename, e))

    return jpeg_df


def correct_EXIF_GPS_coordinates(jpeg_df, calculate_gps_corrections, cache_file):
    """ The original jpeg_df should contain all the JPG filenames - and associated EXIF data
    """
    # The following are common indicates I've seen that the latitude/longitude is wrong.
    jpeg_df2 = jpeg_df.loc[
        (jpeg_df['gps_altitude'] == 0.0) | (jpeg_df['latitude'] is None) | (jpeg_df['longitude'] is None)]

    if calculate_gps_corrections:
        # This is a relatively slow operation - reading lots of GPX files and attempting to find GPS coordinates at
        # the same date/time as EXIF's exposure time
        jpeg_df3, corrected_count = attempt_to_recover_GPS_coords_in_images(jpeg_df2, args.tracks, args.debug,
                                                                            show_changes=(args.correctgps >= 2))
        # jpeg_df3 contains the corrections found from the GPX files.
    else:
        jpeg_df3 = pd.DataFrame()  # empty
        corrected_count = 0

    if cache_file is not None:
        cache_df = deal_with_cache_corrections_file(jpeg_df3, cache_file)  # Will read the previous cache-file (if any),
                                                                        # and update that with the data from jpeg_df3
                                                                        # (i.e. corrections from above) cache_df
                                                                        # contains all the corrections - whether
                                                                        # calculated above or loaded from CSV
        jpeg_df.update(cache_df)  # Update the original with the corrections
    return jpeg_df, corrected_count


class LiftHelper:
    def __init__(self, name, the_args):
        self.name = name
        self.the_args_dict = StringToDict(the_args)
        self.feature_group = folium.FeatureGroup(name, overlay=True)
        self.feature_group.add_to(folium_map)

    def AddSkiLift(self, liftname, lower_lat, lower_lon, upper_lat, upper_lon):
        folium.PolyLine([(lower_lat, lower_lon), (upper_lat, upper_lon)],
                        **self.the_args_dict
                        ).add_to(self.feature_group)


def meters_to_feet(meters):
    if meters is None: return None
    return meters * 3.280839895


def meters_to_US(meters):
    """Convert from meters (which is the units for the gpxpy fields) to more conventional units for the
    USA: Miles, if long enough, otherwise feet.
    """
    if meters is None: return None
    distance_feet = meters_to_feet(meters)
    if distance_feet <= 1000:
        return "{}'".format(int(distance_feet))
    else:
        distance_miles = distance_feet / 5280
        if distance_miles <= 2:
            return '{:.2f} miles'.format(distance_miles)
        elif distance_miles <= 30:
            return '{:.1f} miles'.format(distance_miles)
        else:
            return '{:.0f} miles'.format(distance_miles)


def create_track_popup(gpx_track_segment, name):
    length_2d_meters = gpx_track_segment.length_2d()
    uphill_meters, downhill_meters = gpx_track_segment.get_uphill_downhill()
    start_time, end_time = gpx_track_segment.get_time_bounds()
    min_elevation, max_elevation = gpx_track_segment.get_elevation_extremes()
    duration = end_time - start_time
    duration_cleaned = datetime.timedelta(seconds=round(
        duration.total_seconds()))  # Wow! There's no strftime() equivalent for timedelta. So just trying to get rid of the microseconds

    popup_str = ("<table class='image-list-popup'><caption>{}</caption>\n"
                 "<tr><th>Distance</th><td>{}</td></tr>\n"
                 "<tr><th>Duration</th><td>{}</td></tr>\n"
                 "<tr><th>Elevation Range</th><td>{}'..{}'</td></tr>\n"
                 "<tr><th>Uphill</th><td>{:.0f}'</td></tr>\n"
                 "<tr><th>Date</th><td>{}</td></tr>\n"
                 "</table>\n".format(
        name, meters_to_US(length_2d_meters), duration_cleaned, round(meters_to_feet(min_elevation)),
        round(meters_to_feet(max_elevation)),
        meters_to_feet(uphill_meters), start_time.strftime("%d-%b-%Y")))
    return popup_str


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                        description="""
                        Create an HTML page showing hiking trails with markers
                        for photographs. GPX files (generated by a GPS device)
                        can be loaded the identify a traveler's tracks. These
                        may be overloaded with markers identifying locations
                        where geo-tagged images were taken. Additionally,
                        other indications - such as known hiking paths
                        (-fixed) and/or ski-lifts may be displayed, and
                        polygons can be shown to identify different regions
                        (ski-areas, wilderness areas, cities, etc.). This
                        results is an HTML page suitable for viewing with most
                        common browsers.
                        Additional features include a menu that allows
                        selection of the back-ground maps and selections of
                        which tracks and/or date-times are to be displayed.
                        """,
                        epilog="""
                        This implementation uses the folium python library -
                        which uses the leaflet.js (javascript) library.
                        Therefore, some of the argument support is dependent
                        on those libraries.
                        
                        Further notes:
                        <args-see-descr> indicates a specific format which
                            various arguments may be passed to some of the
                            underlying 'folium' library routines. And example:
                                    "{'color': 'red', 'weight': 5}"
                            where the keywords (color and weight) are supported
                            in the underlying folium routine. Unfortunately,
                            I'm not aware of robust documentation as to what
                            keys are supported or their legal values. Also, be
                            aware that some of the argument-names change
                            slightly; such as the 'dashArray' argument in the
                            leaflet.js library is supported as 'dash_array' in
                            folium - and therefore this program. Note that the
                            usages of the double (") and single (') quotes.
                        gps-correction: Although intended for allowing faulty
                            GPS location information in an image's EXIF, this
                            is also useful for images without such information
                            but contain accurate time-stamps.
                            Some 'bad' GPS coordinates can be recognized from
                            their values - such as missing GPS fields or images
                            where the GPS altitude is reported as 0.0.
                            In those cases, the software can attempt to apply
                            correct GPS locations if the hiking tracks were
                            using a different GPS device (perhaps one is from
                            a hiking partner), and the hiking tracks were
                            reliable and contain embedded time-stamps. Those
                            times can then be matched up against the JPG with
                            the faulty GPS information. Since this 'correction'
                            process can take significant extra processing time,
                            the corrections can be cached in a CSV file.
                        Regarding the different types of indications: 
                            -fixed is intended to show hiking trails, roads and
                                so on. The files with that data is not expected
                                to contain date and time information. These
                                paths are NOT checked against the -include and
                                -exclude regions. Typically loaded from a GPX file.
                            -tracks is intended to show the actual path taken by a
                                traveler (hiker, biker, walker, kayaker, skier,
                                etc.) This data will normally contain date/time
                                information, although that is usually not very
                                important to this program (except for gps-
                                correction). Tracks are checked against
                                -include and -exclude regions. Typically loaded
                                from a GPX file.
                            -polygon is somewhat similar to -fixed but intended
                                to show enclosed areas - and so is created as a
                                polygon rather than a polyline. This means the
                                -polygon has an interior - and additional
                                attributes (such as fill_color) may be applied.
                                Typically loaded from a GPX file.
                            -lifts are little more than lines - identified by
                                the two latitude,longitude of each of the two
                                end-points. These are loaded from a CSV file.
                        -include and -exclude help to filter the input data,
                            both for efficiency and for privacy. Anvimage-
                            marker or a point in a -tracks will NOT be
                            displayed if it is any -exclude'd region. If there
                            is at least one -include region, then a image-
                            marker or a point in a -tracks will be shown only
                            if it is in at least one of the -include region(s).
                            Note that -exclude has priority over -include. Both
                            -include and -exclude are identified by a "bounding
                            box" - which is a rectangular region (if one
                            assumes longitudinal lines are parallel). A
                            bounding box can be identified in one of two ways:
                                3 values: latitude,longitude,edge-length: the
                                    latitude and longitude identify the center
                                    of the box.
                                4 values: min-latitude,min-longitude,
                                    max-latitude,max-longitude. I.e. the lower
                                    left and upper right corners of the box.
                        -pm_group - if there are a lot of images associated
                            with a narrow range of latitude/longitude
                            coordinates, then the resulting number of image-
                            markers on the map would be too crowded - and some
                            markers would be very difficult to select since
                            folium would consider them below others. So the
                            system may group several images into a single
                            marker; the marker's popup is then used to select
                            the desired image. This grouping is done by
                            creating a virtual grid of a certain size; images
                            at the same grid locations are grouped into a
                            single marker. (Note: the marker is displayed not
                            at the center of that grid location, but at the
                            geographic center of all the images in that marker.)
                        """
                                         )
    arg_parser.add_argument("-debug", "--debug", type=int, action="store", nargs='?', const=1,
                            help="set debug variable (0=off, 1-10 is typical range)", default=0)
    arg_parser.add_argument("-profile", "--profile", action="store_true", help="Enable run-time profiling.",
                            default=False)
    arg_parser.add_argument("-outfile", "--outfile", type=str, action="append",
                            help="Name of file to be created with the generated HTML.")

    arg_parser.add_argument("-tracks", "--tracks",
                            help="location of the GPX file(s) identify hiking tracks (path hiked): <file-or-directory> [<args-see-descr>']",
                            nargs="+", action="append", type=str)

    arg_parser.add_argument("-photosdir", "--photosdir", type=str, action="append",
                            help="Directory to scan for image files: <directory-name> <href-such-as-url> [<args-see-descr>]",
                            nargs="+", default=None)

    arg_parser.add_argument("-pm_group", "--pm_group", type=float,
                            help="Photo-marker grouping resolution in feet. Creates a grid of given resolution - groups images at same grid box,",
                            default=150)

    arg_parser.add_argument("-fixed", "--fixed", type=str,
                            help="Show a fixed path on the map. Positional arguments are: <gpx-filename> [<label> [<args-see-descr> [<popup-text>]]]",
                            nargs="+", action='append')

    arg_parser.add_argument("-polygon", "--polygon", type=str,
                            help="Draw a polygon - perimeter. To outline a region/boundary. Positional arguments are: <gpx-filename> [<label> [<args-see-descr> [<popup-text>]]]]",
                            nargs="+", action='append')

    arg_parser.add_argument("-lifts", "--lifts", type=str,
                            help="CSV file identifying GPS coordinates of lower and upper terminals of ski-lifts. CSV headers should be: Lift,LowerTermLat,LowerTermLon,UpperTermLat,UpperTermLon. Positional command-line arguments are <csv-filename> [args-see-descr>]",
                            nargs="+", action='append')

    arg_parser.add_argument("-include", "--include", type=float,
                            help="Define a region which each GPS point must be in to be included - for photos and tracks."
                                 " Three decimal (floating point) values as: <latitude> <longitude> <edge-length-miles>"
                                 " Such that <latitude> and <longitude> identify the center point of a rectange - and <edge-length> indicates that length of each edge"
                                 " Default is to include all points",
                            nargs=3, action="append", default=None)

    arg_parser.add_argument("-exclude", "--exclude", type=float,
                            help="Define a region which each GPS will be discarded for photos and tracks."
                                 " Similar to --include: Three decimal (floating point) values as: <latitude> <longitude> <edge-length-miles>"
                                 " Note - --exclude has priority over --include: points in both an --include and an --exclude region will be excluded."
                                 " Default is to exclude no points",
                            nargs='+', action="append", default=None)

    arg_parser.add_argument("-showbbox", "--showbbox", type=int, action="store", nargs='?', const=1,
                            help="Show the bounding-boxes of --include and --exclude. 0=disable, 1=enable, 2=available on layers menu",
                            default=0)
    arg_parser.add_argument("-showrejections", "--showrejections", type=int, action="store", nargs='?', const=1,
                            help="Show the points rejected due to bounding-boxes.", default=0)
    arg_parser.add_argument("-correctgps", "--correctgps", type=int, action="store", nargs='?', const=1,
                            help="Attempt to correct 'bad' GPS coordinates in JPEG files using date/time of tracking (GPX) files. 0=disable, 1=enable, 2=show changes",
                            default=0)
    arg_parser.add_argument("-cachecorrections", "--cachecorrections", type=str, action="store", nargs='?',
                            help="CSV file save/reload corrections from -correctgps.")

    args = arg_parser.parse_args()

    if args.debug >= 1:
        print("folium version={}".format(folium.__version__))
        print("pandas version={}".format(pd.__version__))
        print("---------------------")
        print("debug({}): {}".format(type(args.debug), args.debug))
        print("profile({}): {}".format(type(args.profile), args.profile))
        print("tracks({}): {}".format(type(args.tracks), args.tracks))
        print("outfile({}): {}".format(type(args.outfile), args.outfile))
        print("photosdir({}): {}".format(type(args.photosdir), args.photosdir))
        print("pm_group({}): {}".format(type(args.pm_group), args.pm_group))
        print("fixed({}): {}".format(type(args.fixed), args.fixed))
        print("polygon({}): {}".format(type(args.polygon), args.polygon))
        print("include({}): {}".format(type(args.include), args.include))
        print("exclude({}): {}".format(type(args.exclude), args.exclude))
        print("showbbox({}): {}".format(type(args.showbbox), args.showbbox))
        print("showrejections({}): {}".format(type(args.showrejections), args.showrejections))
        print("correctgps({}): {}".format(type(args.correctgps), args.correctgps))
        print("cachecorrections({}): {}".format(type(args.cachecorrections), args.cachecorrections))
        print("lifts({}): {}".format(type(args.lifts), args.lifts))
        print("---------------------")

    round_resolution_degrees = feet_to_degrees(args.pm_group)

    folium_map = folium.Map(tiles="OpenStreetMap", control_scale=False)
    # folium.TileLayer('OpenStreetMap', attr="Open Street Map", name="Open Street Map").add_to(folium_map)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
                     attr="Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC",
                     name='Nat Geo Map').add_to(folium_map)

    folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr="terrain-bcg",
                     name='Stamen Terrain Map').add_to(folium_map)
    folium.TileLayer('Stamen Toner', attr="Stamen Toner", name="Stamen Toner").add_to(folium_map)
    folium.TileLayer('Stamen Watercolor', attr="Stamen Watercolor", name="Stamen Watercolor").add_to(folium_map)

    t1 = get_time()
    if args.profile: print("Program startup and map initialization took {:.6f}s".format(t1 - time_program_start))

    include_boxes = []  # List of bounding boxes - each list entry is a tuple with 4 points - the GPS coordinates of SW and NE i.e. (min lat, min lon, max lat, max lon)
    exclude_boxes = []  # List of bounding boxes - each list entry is a tuple with 4 points - the GPS coordinates of SW and NE i.e. (min lat, min lon, max lat, max lon)
    if (args.include is not None) or (args.exclude is not None):
        if args.include is not None:
            for include_info in args.include:
                bbox = CreateBBox(include_info)
                include_boxes.append(bbox)
        if args.debug >= 3: print("include_boxes={}".format(include_boxes))
        if args.exclude is not None:
            for exclude_info in args.exclude:
                bbox = CreateBBox(exclude_info)
                exclude_boxes.append(bbox)
        if args.debug >= 3: print("exclude_boxes={}".format(exclude_boxes))
    t2 = get_time()
    if args.profile: print("Decode the include/exclude boxes took {:.6f}s".format(t2 - t1))

    # Read EXIF data - create a DataFrame
    # DVO HELP - should support reading from mulitple directories - i.e. accumulating in the exif_dict - the following code has a loop - but only the last iteration is used (i.e. that's a bug)
    if (args.photosdir is not None) and (len(args.photosdir) >= 1):
        jpeg_df = pd.DataFrame()
        for stuff in args.photosdir:
            t2a = get_time()
            directory_name = stuff[0]
            the_photos_href = stuff[1] if (len(stuff) >= 2) and (stuff[1] is not None) else ""
            the_photos_args_string = stuff[2] if (len(stuff) >= 3) and (stuff[1] is not None) else ""
            the_photos_args_dict = StringToDict(the_photos_args_string)

            coord_df = get_EXIF_from_directory(directory_name, the_photos_href, debug=False)

            # create dataframe from extracted jpeg/EXIF data
            jpeg_df = pd.concat([jpeg_df, coord_df])
            t2b = get_time()
            if args.profile: print(
                "Reading EXIF data from folder ({}) and creating DataFrame - took {:.6f}s".format(directory_name,
                                                                                                  t2b - t2a))
        jpeg_df.set_index('filename', inplace=True, drop=False)

    t3 = get_time()
    if args.profile: print("Reading EXIF data and creating DataFrame - took {:.6f}s".format(t3 - t2))

    if (args.correctgps >= 1) or (args.cachecorrections is not None):
        jpeg_df, corrected_count = correct_EXIF_GPS_coordinates(jpeg_df, args.correctgps >= 1, args.cachecorrections)

    t4 = get_time()
    if (args.correctgps >= 1) and (args.debug >= 1): print(
        "Corrected {} GPS coordinate{} from image files. Took {:.6f}s".format(corrected_count,
                                                                              "" if corrected_count == 1 else "s",
                                                                              t4 - t3))

    t5 = get_time()

    if args.polygon is not None:
        for stuff in args.polygon:
            filename = stuff[0]
            name = stuff[1] if (len(stuff) >= 2) and (stuff[1] is not None) else os.path.basename(filename)
            the_args_string = stuff[2] if (len(stuff) >= 3) and (stuff[2] is not None) else ""
            popup = stuff[3] if (len(stuff) >= 4) else None
            feature_group = folium.FeatureGroup(name, overlay=True)
            list_of_tracks, first_time = read_gpx_file(filename, include_datetime=False, profile=args.profile)
            the_args_dict = StringToDict(the_args_string)
            for track_pair in list_of_tracks:
                folium.Polygon(track_pair[1],
                               **the_args_dict,
                               tooltip=name,
                               popup=popup).add_to(feature_group)
            feature_group.add_to(folium_map)
    t5a = get_time()
    if args.profile: print("Mapping Polygons took {:.6f}s".format(t5a - t5))

    if args.lifts is not None:
        for stuff in args.lifts:
            filename = stuff[0]
            the_name = stuff[1] if (len(stuff) >= 2) and (stuff[1] is not None) else "lifts"
            the_args = stuff[2] if (len(stuff) >= 3) and (stuff[2] is not None) else ""

            if os.path.isfile(filename):
                try:
                    lifts_df = pd.read_csv(filename)
                except IOError as e:
                    print("ERROR: error while attempting to read {}: {}".format(filename, e))
                    exit(1)
            else:
                print("ERROR: File not found: {}".format(filename))
                exit(1)

            lifts = LiftHelper(the_name, the_args)
            [lifts.AddSkiLift(liftname, lower_lat, lower_lon, upper_lat, upper_lon) for
             liftname, lower_lat, lower_lon, upper_lat, upper_lon in zip(
                lifts_df['Lift'], lifts_df['LowerTermLat'], lifts_df['LowerTermLon'], lifts_df['UpperTermLat'],
                lifts_df['UpperTermLon'])]

    t5b = get_time()
    if args.profile: print("Lifts took {:.6f}s".format(t5b - t5a))

    if args.debug >= 2: print("Before dealing with Fixed Routes")
    # Deal with the pre-defined Fixed Routes
    if args.fixed is not None:
        for stuff in args.fixed:
            filename = stuff[0]
            name = stuff[1] if (len(stuff) >= 2) and (stuff[1] is not None) else os.path.basename(filename)
            the_args = stuff[2] if (len(stuff) >= 3) and (stuff[2] is not None) else ""
            popup = stuff[3] if (len(stuff) >= 4) else None
            feature_group = folium.FeatureGroup(name, overlay=True)
            list_of_tracks, first_time = read_gpx_file(filename, profile=args.profile)
            the_args_dict = StringToDict(the_args)
            for track_pair in list_of_tracks:
                folium.PolyLine(track_pair[1],
                                **the_args_dict,
                                tooltip=name,
                                popup=popup).add_to(feature_group)
            feature_group.add_to(folium_map)
    t6 = get_time()
    if args.profile: print("Mapping Fixed Routes took {:.6f}s".format(t6 - t5b))

    photo_group = folium.FeatureGroup("Photographs", control=True, show=True)
    folium_map.add_child(photo_group)

    t7 = get_time()

    if round_resolution_degrees != 0.0:
        # https://stackoverflow.com/questions/26133538/round-a-single-column-in-pandas
        if "latitude" in jpeg_df:
            jpeg_df.loc[:, 'lat_round'] = jpeg_df['latitude'].apply(
                lambda value: round(value / round_resolution_degrees) * round_resolution_degrees if (
                        (value is not None) and not math.isnan(value)) else None)
            # Weird - the above can leave some nan values in the "lat_round" and "lon_round" columns - I tried to debug but didn't make progress. Want to handle here rather than downstream.
            # jpeg_df['lat_round'].replace( { np.nan : None }, inplace=True) # This also doesn't work for me
        if "longitude" in jpeg_df:
            jpeg_df.loc[:, 'lon_round'] = jpeg_df['longitude'].apply(
                lambda value: round(value / round_resolution_degrees) * round_resolution_degrees if (
                        (value is not None) and not math.isnan(value)) else None)

        if args.debug >= 3:
            try:
                [print(
                    "Original: ({:8.6f},{:8.6f})   Rounded: ({:8.6f},{:8.6f}, {}))".format(lat, lon, lat_rnd, lon_rnd,
                                                                                           (lat == lat_rnd) and (
                                                                                                   lon == lon_rnd)))
                    for lat, lon, lat_rnd, lon_rnd in zip(
                    jpeg_df['latitude'], jpeg_df['longitude'], jpeg_df['lat_round'], jpeg_df['lon_round'])]
            except BaseException as e:
                print("Exception encountered in debug code: {}. {}".format(type(e), e))

        # Sort by lat_round and then lon_round
        if ("lat_round" in jpeg_df) and ("lon_round" in jpeg_df):
            jpeg_df.sort_values(by=['lat_round', 'lon_round'], inplace=True)

        if args.debug >= 3:
            try:
                [print(
                    "Original: ({:8.6f},{:8.6f})   Rounded and Sorted: ({:8.6f},{:8.6f}, {})".format(lat, lon, lat_rnd,
                                                                                                     lon_rnd, (
                                                                                                             lat == lat_rnd) and (
                                                                                                             lon == lon_rnd)))
                    for lat, lon, lat_rnd, lon_rnd in zip(
                    jpeg_df['latitude'], jpeg_df['longitude'], jpeg_df['lat_round'], jpeg_df['lon_round'])]
            except BaseException as e:
                print("WARNING exception caught in debug code: {}, {}".format(type(e), e))

    t8 = get_time()
    if args.profile: print("Sorting by latitude and longitude took {:.6f}s".format(t8 - t7))

    dbl_filtered_df = jpeg_df.groupby(["lat_round", "lon_round"])
    for lon_round, thing2 in dbl_filtered_df:
        thing3 = thing2.copy()  # This "fixes" the SettingWithCopyWarning mentioned elsewhere in this file.
        create_photo_marker(thing3, include_boxes, exclude_boxes, args.showrejections, the_photos_args_dict)

    t9 = get_time()
    if args.debug >= 1: print("Grouping by latitude and longitude {:.6f}s".format(t9 - t8))

    for stuff in args.tracks:  # DVO TODO - restructure so that args.tracks can be a list of files and/or of directories (now assumes directories)
        t9a = get_time()
        pathname = stuff[0] if (len(stuff) >= 1) and (stuff[0] is not None) else ""
        feature_name = stuff[1] if (len(stuff) >= 2) and (stuff[1] is not None) else ""
        the_args_string = stuff[2] if (len(stuff) >= 3) and (stuff[2] is not None) else ""
        the_args_dict = StringToDict(the_args_string)
        file_list = [os.path.join(pathname, f) for f in sorted(os.listdir(pathname), reverse=True)] if os.path.isdir(
            pathname) else [pathname]
        tracking_group = folium.FeatureGroup(feature_name, control=True, show=True)
        tracking_group.add_to(folium_map)
        for filename in file_list:
            if filename.endswith(".gpx"):
                if debug_level >= 1:
                    print("GPX file={}".format(filename))
                (dir_part, file_part) = os.path.split(filename)
                (root_part, ext_part) = os.path.splitext(file_part)
                the_label = os.path.basename(root_part)
                list_of_tracks, first_time = read_gpx_file(filename, profile=args.profile)
                if debug_level >= 10:
                    print("type(list_of_tracks)={}".format(type(list_of_tracks)))
                    print("list_of_tracks=", list_of_tracks)

                formatted_timestamp = TzHelper.as_local(first_time).strftime("%Y%m%d") if first_time is not None else "Undated"
                checkbox_datestamps.add(formatted_timestamp)

                for track_tuple in list_of_tracks:
                    adjusted_list_of_points = remove_excluded_points(track_tuple[1], exclude_boxes, args.showrejections >= 1)
                    polyline = folium.PolyLine(adjusted_list_of_points,
                                               **the_args_dict,
                                               tooltip=the_label,
                                               # Tried lots of variations - but couldn't get the className stuff to pass information into the generated .html file
                                               dash_array="class_" + formatted_timestamp,
                                               # This is a kluge - apparently folium doesn't support class_name="..." (documentation
                                               # implies it should), so I'm picking a little-used attribute (dashArray) that is supported and hijacking it.
                                               # Relying on a post-processing step in the generated .html file to change the generated "dashArray" to the desired "className"
                                               # There's another near-identical kluge elsewhere in this file.
                                               popup=create_track_popup(track_tuple[2], the_label)
                                               )
                    polyline.add_to(tracking_group)
        tracking_group.add_to(folium_map)
        if args.profile: print(
            "Processing GPX files ({}) and marking routes on map - took {:.6f}s".format(pathname, get_time() - t9a))
    t10 = get_time()
    if args.profile and (len(args.tracks) > 1): print(
        "Processing all GPX fles and marking routes on map - took {:.6f}s".format(t10 - t9))

    if (args.showbbox > 0) and ((args.include is not None) or (args.exclude is not None)):
        bbox_group = folium.FeatureGroup("Bounding Boxes", control=True, show=(args.showbbox == 1))
        bbox_group.add_to(folium_map)
        folium_map.add_child(bbox_group)
        for bbox in include_boxes:
            the_rectangle = folium.Rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], color="green")
            the_rectangle.add_to(bbox_group)
        for bbox in exclude_boxes:
            the_rectangle = folium.Rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], color="red")
            the_rectangle.add_to(bbox_group)
    t11 = get_time()
    if args.profile: print("Decode and draw the include/exclude boxes took {:.6f}s".format(t11 - t10))

    folium_map.fit_bounds(
        [(Overall_min_latitude, Overall_min_longitude), (Overall_max_latitude, Overall_max_longitude)])
    folium.LayerControl().add_to(folium_map)
    t12 = get_time()
    if args.profile: print("fit_bounds() and LayerControl().add_to(...) took {:.6f}s".format(t12 - t11))

    # Create the DateTime checkboxes menu
    checkboxes = '<form id="date_menu_form_id" onsubmit="return false;"><fieldset><legend align="center">Date Selector</legend>\n' \
                 + '<table class="image-list-popup"><caption>For Tracks and Photos which are enabled in the Layers menu.</caption>\n'
    cell_count = 0
    for datestamp in sorted(checkbox_datestamps, reverse=True):
        cell_count += 1
        pretty_datestamp = datetime.datetime.strptime(datestamp, "%Y%m%d").strftime("%d-%b-%Y")
        if ((cell_count - 1) % 4) == 0:
            checkboxes += "<tr>"
        checkboxes += '<td><label><input type="checkbox" name="{}" value="{}" checked id="{}" onclick="date_menu_checkbox_toggle({})">{}</label></td>\n'.format(
            datestamp, datestamp, datestamp, datestamp, pretty_datestamp)
        if ((cell_count % 4) == 0):
            checkboxes += "</tr>\n"
    if (cell_count % 4):
        while (cell_count % 4):
            checkboxes += "<td></td>"
            cell_count += 1
        checkboxes += "</tr>\n"

    checkboxes += '<tr><td colspan="2"><button id="datestamp_button">Uncheck All</button></td>' \
                  + '<td colspan="2"><button id="toggle_datestamp_button">Toggle All</button></td></tr>\n' \
                  + '</table></fieldset></form>'

    # A lot of pass-through code directly into the output HTML file.
    folium_map.get_root().html.add_child(folium.Element("""
<style>
.calendar_icon {{
        font-size: 24px;
        opacity: 0.4;
}}
.date_menu {{
        display: none;
}}
.date_menu_icon:hover + .date_menu {{
        display: block;
}}
.date_menu:hover {{
        display: block;
}}
</style>
 <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
        <div class="date_menu_icon"; style="position: fixed;
         top: 50px; right: 10px; width: 36px; height: 36px;
         background-color:white; border:0; border-radius: 5px; padding: 7px; box-shadow: 0px 2px 4px gray; z-index: 900;">
                <span class="glyphicon glyphicon-calendar calendar_icon"></span>
        </div>
        <div class="date_menu";
        style="position: fixed;
         top: 50px; right: 10px;
         background-color:white; border:0; border-radius: 5px; padding: 7px; box-shadow: 1px 2px 2px lightgray; z-index: 901;">
        {}
        </div>
	<script>
        ////////////////////////////////

        const check_all = "Check All";
        const uncheck_all = "Uncheck All";
        const ds_btn = document.getElementById("datestamp_button");
        checked_value = true;
        visibility_value = "visible";
        ds_btn.addEventListener('click', () => {{
                if (ds_btn.innerText === check_all) {{
                    ds_btn.innerText = uncheck_all;
                    checked_value = true;
                    visibility_value = "visible";
                }} else if (ds_btn.innerText == uncheck_all) {{
                    ds_btn.innerText = check_all;
                    checked_value = false;
                    visibility_value = "hidden";
                }}
                form = document.getElementById('date_menu_form_id');
                Array.from(form.elements).forEach(element => {{
                    if (0) {{
                        console.log('line 867: element=', element);
                        console.log('line 868:     name=', element.name );
                        console.log('line 869:     type=', element.type );
                        console.log('line 870:     value=', element.value );
                        console.log('line 871:     checked=', element.checked );
                        console.log('line 872:     tag=', element.tag );
                        console.log('line 873:     text=', element.text );
                        console.log('line 874:     id=', element.id );
                        console.log('line 875:     innerText=', element.innerText );
                        console.log('line 876:     innerHTML=', element.innerHTML );
                    }}
                    if (element.type === "checkbox") {{
                        SetAllElementsInCssClass(element.name, "visibility", visibility_value );
                        element.checked = checked_value;
                    }}
                }});
                return false;
        }});


        const toggle_button = document.getElementById("toggle_datestamp_button");
        toggle_button.addEventListener('click', () => {{
                form = document.getElementById('date_menu_form_id');
                Array.from(form.elements).forEach(element => {{
                    if (element.type === "checkbox") {{
                        if (element.checked) {{
                            SetAllElementsInCssClass(element.name, "visibility", "hidden" );
                            element.checked = false;
                        }} else {{
                            SetAllElementsInCssClass(element.name, "visibility", "visible" );
                            element.checked = true;
                        }}
                    }}
                }});
        return false;
        }});

        function date_menu_checkbox_toggle( classname) {{
            if (document.getElementById(classname).checked) {{
                SetAllElementsInCssClass(classname,"visibility","visible");
            }} else {{
                SetAllElementsInCssClass(classname,"visibility","hidden");
            }}
        }}
        function SetAllElementsInCssClass(classname, property_str, value) {{
            //console.log("In SetAllElementsInCssClass( class=", classname, ", property=", property_str, ", value=", value, ")" );
            the_elements = document.getElementsByClassName( classname );
            //console.log("the_elements length=", the_elements.length );
            for (var ii=0; ii<the_elements.length; ii++) {{
                //console.log("element[",ii,"]=", the_elements[ii] );
                the_elements[ii].style.setProperty( property_str, value );
            }}
        }}
	</script>
    """.format(checkboxes)))

    folium.folium._default_css.append(('leaflet_overloaded_css', 'http://kokanee.mynetgear.com/dvo/hiking_map.css'))
    t13 = get_time()
    if args.profile: print(
        "Creating date/time checkbox menu - and transfering HTML/CSS/Javascript code to {:.6f}s".format(t13 - t12))

    for outfile in args.outfile:  # I don't understand: the 2nd output file (if any) may have added a call to remove() the feature-group for the bounding boxes. But the
        # observable effect is that the HTML map starts without any background map selected.
        print("Saving {}".format(outfile))
        folium_map.save(outfile)
    t14 = get_time()
    if args.profile: print("Saving {} took {:.6f}s".format(args.outfile, t14 - t13))
    # webbrowser.open(args.outfile)
    t15 = get_time()
    # if args.profile: print("After webbrowser.open({}) took {:.6f}s".format(args.outfile, t15-t14))
    if args.profile: print("Total execution time: {:.6f}s".format(t15 - time_program_start))

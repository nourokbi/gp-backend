import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import base64
import io

import firebase_admin
from firebase_admin import db, credentials, storage
from firebase_admin.exceptions import FirebaseError

import numpy as np
import uuid
import netCDF4 as nc
import cftime
from datetime import datetime

from flask import Flask, request, jsonify
from flask_restful import Api

from shapely.geometry import Polygon

app = Flask(__name__)

api = Api(app)

# Fetch the service account key Json file content
cred = credentials.Certificate('graduation-beef5-firebase-adminsdk-afqzq-dd4a2248ce.json')

# initialize the app with a service account, granting admin privileges

config = {
    "apiKey": "AIzaSyAW0rmbkCChp9Q305VcRtoQvgWlY4e8VG0",
    "authDomain": "graduation-beef5.firebaseapp.com",
    "databaseURL": "https://graduation-beef5-default-rtdb.firebaseio.com",
    "projectId": "graduation-beef5",
    "storageBucket": "graduation-beef5.appspot.com",
    "messagingSenderId": "446252346321",
    "appId": "1:446252346321:web:ce8d1203fc81cee1d20ef0",
    "measurementId": "G-3V65JWN3GJ"
}
firebase_admin.initialize_app(cred, config)


def get_min_max_dates(file_path):
    """
    Extracts the minimum and maximum dates from a NetCDF file.

    Args:
    - file_path (str): The path to the NetCDF file.

    Returns:
    - min_date (str): The minimum date in 'YYYY-MM-DD' format.
    - max_date (str): The maximum date in 'YYYY-MM-DD' format.
    # Example usage
        file_path = 'netcdf_files/ndays_maxt.gt.25_RCP85.nc'
        min_date, max_date = get_min_max_dates(file_path)
        print(f"Minimum Date: {min_date}")
        print(f"Maximum Date: {max_date}")
    """
    dataset = nc.Dataset(file_path)

    try:
        # Get the 'time' variable
        time_variable = dataset.variables['time']

        # Extract the minimum and maximum values
        min_date = time_variable[:].min()
        max_date = time_variable[:].max()

        # Convert numerical dates to datetime objects using cftime
        min_date_datetime = cftime.num2date(min_date, units=time_variable.units, calendar=time_variable.calendar)
        max_date_datetime = cftime.num2date(max_date, units=time_variable.units, calendar=time_variable.calendar)

        # Format dates as 'YYYY-MM-DD'
        min_date_formatted = min_date_datetime.strftime("%Y-%m-%d")
        max_date_formatted = max_date_datetime.strftime("%Y-%m-%d")

        return min_date_formatted, max_date_formatted

    finally:
        # Close the NetCDF file
        dataset.close()


def get_available_indexes_according_to_type(t):
    # if t == 'temp':
    #     return [
    #         'FD', 'SU', 'ID', 'TR', 'GSL', 'TXx', 'TNx', 'TXn', 'TNn', 'TN10p', 'TX10p',
    #         'TN90p', 'TX90p', 'WSDI', 'CSDI', 'DTR', 'ETR', 'CDDcoldn', 'GDDgrown',
    #         'HDDheatn', 'TMge5', 'TMlt5', 'TMge10', 'TMlt10', 'TMm', 'TXm', 'TNm',
    #         'TXge30', 'TXge35', 'TXgt50p', 'TNlt2', 'TNltm2', 'TNltm20', 'TXbdTNbd'
    #     ]
    # elif t == 'pr':
    #     return ['Rx1day', 'Rx5day', 'SPI', 'SPEI', 'SDII', 'R10mm', 'R20mm', 'Rnnmm', 'CDD', 'CWD', 'R95p', 'R99p',
    #             'R95pTOT', 'R99pTOT', 'PRCPTOT']
    # elif t == 'lone':
    #     return []
    # else:
    #     return ['HWN', 'HWF', 'HWD', 'HWM', 'HWA', 'CWN_ECF', 'CWF_ECF', 'CWD_ECF', 'CWM_ECF', 'CWA_ECF']
    #

    if t == 'min_temp':
        # FD, TR, TNx, TNn, TN10p, TN90p, TNlt2, TNltm2, TNltm20,
        return [
            'FD', 'TR', 'TNx', 'TNn', 'TN10p', 'TN90p', 'TNlt2', 'TNltm2', 'TNltm20'
        ]
    elif t == 'max_temp':
        # SU, ID, TXx, TXn, TX10p, TX90p, WSDI, CSDI, TXm, TNm, TXge30, TXge35, TXgt50p,
        return [
            'SU', 'ID', 'TXx', 'TXn', 'TX10p', 'TX90p', 'WSDI', 'CSDI', 'TXm', 'TNm', 'TXge30', 'TXge35', 'TXgt50p'
        ]
    # for both min and max temp: DTR, ETR, TXbdTNbd
    elif t == 'min_max_temp':
        return [
            'DTR', 'ETR', 'TXbdTNbd',  # needs both min and max
            'FD', 'TR', 'TNx', 'TNn', 'TN10p', 'TN90p', 'TNlt2', 'TNltm2', 'TNltm20', # needs min
            'SU', 'ID', 'TXx', 'TXn', 'TX10p', 'TX90p', 'WSDI', 'CSDI', 'TXm', 'TNm', 'TXge30', 'TXge35', 'TXgt50p' # needs max
        ]
    elif t == 'min_mean_temp':
        return [
            'FD', 'TR', 'TNx', 'TNn', 'TN10p', 'TN90p', 'TNlt2', 'TNltm2', 'TNltm20', # needs min
            'GSL', 'CDDcoldn', 'GDDgrown', 'HDDheatn', 'TMge5', 'TMlt5', 'TMge10', 'TMlt10', 'TMm'
        ]
    elif t == 'max_mean_temp':
        return [
            'SU', 'ID', 'TXx', 'TXn', 'TX10p', 'TX90p', 'WSDI', 'CSDI', 'TXm', 'TNm', 'TXge30', 'TXge35', 'TXgt50p', # needs max
            'GSL', 'CDDcoldn', 'GDDgrown', 'HDDheatn', 'TMge5', 'TMlt5', 'TMge10', 'TMlt10', 'TMm'
        ]
    elif t == 'mean_temp':
        # GSL, CDDcoldn, GDDgrown, HDDheatn, TMge5, TMlt5, TMge10, TMlt10, TMm,
        return [
            'GSL', 'CDDcoldn', 'GDDgrown', 'HDDheatn', 'TMge5', 'TMlt5', 'TMge10', 'TMlt10', 'TMm'
        ]
    elif t == 'pr':
        # Rx1day, Rx5day, SDII, R10mm, R20mm, Rnnmm, CDD, CWD, R95p, R99p, R95pTOT, R99pTOT, PRCPTOT
        return [
            'Rx1day', 'Rx5day', 'SDII', 'R10mm', 'R20mm', 'Rnnmm', 'CDD', 'CWD', 'R95p', 'R99p', 'R95pTOT', 'R99pTOT',
            'PRCPTOT'
        ]
    else:
        # a lone index
        return [t]


def create_dataset(path_local, name, type, access, view, description):
    available_indexes = get_available_indexes_according_to_type(type)
    # Extracts the minimum and maximum dates from a NetCDF file.
    min_date, max_date = get_min_max_dates(path_local)

    dataset_id = str(uuid.uuid4())
    file_path_on_cloud = f'datasets/{access}/{dataset_id}'
    return {
        'id': dataset_id,
        'name': name,
        'file_path_on_cloud': file_path_on_cloud,
        'type': type,
        'description': description,
        'available_indexes': available_indexes,
        'var_name': type,
        'min_date': min_date,
        'max_date': max_date,
        'access': access,
        'view': view
    }


def send_dataset_to_firebase(dataset, path_local):
    bucket = storage.bucket()
    path_on_cloud = dataset['file_path_on_cloud']
    blob = bucket.blob(path_on_cloud)
    blob.upload_from_filename(path_local)
    ref = db.reference(f'datasets/{dataset["access"]}/{dataset["id"]}')
    ref.set(dataset)

def upload_dataset(path_local, name, type, access, view, description):
    dataset = create_dataset(path_local, name, type, access, view, description)
    send_dataset_to_firebase(dataset, path_local)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset_endpoint():
    # Get data from the request
    data = request.get_json()

    # Extract parameters from the data
    path_local = data.get('path_local')
    name = data.get('name')
    type = data.get('type')
    access = data.get('access')
    view = data.get('view')
    description = data.get('description')

    # Upload the dataset
    upload_dataset(path_local, name, type, access, view, description)

    return jsonify({"message": "Dataset uploaded successfully"})


@app.route('/update_dataset/<access>/<dataset_id>', methods=['PUT'])
def update_dataset(access, dataset_id):
    try:
        # Parse the request JSON data
        updated_data = request.get_json()

        # Update the dataset in Firebase
        ref = db.reference(f'datasets/{access}/{dataset_id}')
        ref.update(updated_data)

        return jsonify({'message': 'Dataset updated successfully'}), 200
    except FirebaseError as e:
        error_message = f'Error updating dataset: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/delete_dataset/<access>/<dataset_id>', methods=['DELETE'])
def delete_dataset(access, dataset_id):
    try:
        ref = db.reference(f'datasets/{access}/{dataset_id}')
        ref.delete()
        return f'Dataset with ID {dataset_id} deleted successfully from access level {access}.', 200
    except FirebaseError as e:
        error_message = f'Error deleting dataset: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/retrieve_all_datasets/<access>', methods=['GET'])
def retrieve_all_datasets(access):
    try:
        ref = db.reference(f'datasets/{access}')
        datasets_snapshot = ref.get()
        datasets = []
        if datasets_snapshot:
            for dataset_id, dataset_snapshot in datasets_snapshot.items():
                datasets.append(dataset_snapshot)
        return jsonify(datasets), 200
    except FirebaseError as e:
        error_message = f'Error retrieving datasets: {str(e)}'
        return jsonify({'error': error_message}), 500


def create_user(username, email, access):
    return {
        'username': username,
        'email': email,
        'access': access
    }


# Function to encode email address
def encode_email(email):
    return email.replace('.', ',').replace('@', '_at_')


def decode_email(encoded_email):
    return encoded_email.replace(',', '.').replace('_at_', '@')


def add_new_user(username, email, access):
    encoded_email = encode_email(email)
    user = create_user(username, email, access)
    ref = db.reference(f'users/{encoded_email}')
    ref.set(user)


@app.route('/add_new_user', methods=['POST'])
def add_new_user_endpoint():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    access = data.get('access')

    if not username or not email or not access:
        return jsonify({'error': 'Missing required fields (username, email, access)'}), 400

    try:
        add_new_user(username, email, access)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_user/<email>', methods=['PUT'])
def update_user_endpoint(email):
    encoded_email = encode_email(email)
    data = request.json
    username = data.get('username')
    access = data.get('access')

    if not username and not access:
        return jsonify({'error': 'Nothing to update. Provide at least one field (username or access)'}), 400

    try:
        ref = db.reference(f'users/{encoded_email}')
        user = ref.get()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        if username:
            user['username'] = username
        if access:
            user['access'] = access

        ref.update(user)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_user/<email>', methods=['DELETE'])
def delete_user_endpoint(email):
    encoded_email = encode_email(email)
    try:
        ref = db.reference(f'users/{encoded_email}')
        user = ref.get()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        ref.delete()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_user/<email>', methods=['GET'])
def get_user_endpoint(email):
    encoded_email = encode_email(email)
    try:
        ref = db.reference(f'users/{encoded_email}')
        user = ref.get()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify(user), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_all_users', methods=['GET'])
def get_all_users_endpoint():
    try:
        ref = db.reference(f'users')
        users = ref.get()
        if not users:
            return jsonify({'error': 'No Users found'}), 404

        return jsonify(users), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_color_mapping_according_to_type(t):
    if t == 'temp' or 'temperature':
        return 'plasma'
    elif t == 'PR' or 'pr':
        return 'viridis'
    elif t == 'HW' or 'hw':
        return 'magma'
    elif t == 'CW' or 'cw':
        return 'inferno'
    else:
        return 'plasma'


def count_frost_days(data, **kwargs):
    return np.sum(data < 0, **kwargs)


def count_summer_days(data, **kwargs):
    return np.sum(data > 25, **kwargs)


def count_icing_days(data, **kwargs):
    return np.sum(data < 0, **kwargs)


def count_tropical_nights(data, **kwargs):
    return np.sum(data > 20, **kwargs)


def max_daily_max_temp(data, **kwargs):
    return np.max(data, **kwargs)


def max_daily_min_temp(data, **kwargs):
    return np.max(data, **kwargs)


def min_daily_max_temp(data, **kwargs):
    return np.min(data, **kwargs)


def min_daily_min_temp(data, **kwargs):
    return np.min(data, **kwargs)


# Additional Climdex indices functions
def count_growing_season_length(tmean, **kwargs):
    """
    Calculate the growing season length (GSL) for each grid point.
    It counts the number of days between the first period of at least six consecutive days with daily mean temperature above 5°C
    and the first period after July 1 (or January 1 in the southern hemisphere) of six consecutive days with daily mean temperature below 5°C.
    """
    # Check that the input is a 3D array: (time, lat, lon)
    if tmean.ndim != 3:
        raise ValueError("Input data must be a 3D array with dimensions (time, lat, lon)")

    # Initialize the result array with NaNs
    gsl = np.full(tmean.shape[1:], np.nan)

    # Iterate over each grid point
    for lat in range(tmean.shape[1]):
        for lon in range(tmean.shape[2]):
            daily_mean_temp = tmean[:, lat, lon]
            above_5 = daily_mean_temp > 5
            growing_season_started = False
            consecutive_days_above_5 = 0

            for i, day in enumerate(above_5):
                if day:
                    consecutive_days_above_5 += 1
                    if consecutive_days_above_5 >= 6:
                        growing_season_started = True
                else:
                    consecutive_days_above_5 = 0

                if growing_season_started:
                    gsl[lat, lon] = i + 1
                    break

    return gsl


def count_heating_degree_days(data, base_temp=18, **kwargs):
    """
    Calculate Heating Degree Days (HDD).
    It's the sum of the differences between the base temperature (18°C) and the daily mean temperature for all days when the daily mean temperature is below 18°C.
    """
    daily_mean_temp = data
    heating_degree_days = np.sum(np.maximum(base_temp - daily_mean_temp, 0), **kwargs)
    return heating_degree_days


def count_cooling_degree_days(data, base_temp=18, **kwargs):
    """
    Calculate Cooling Degree Days (CDD).
    It's the sum of the differences between the daily mean temperature and the base temperature (18°C) for all days when the daily mean temperature is above 18°C.
    """
    daily_mean_temp = data
    cooling_degree_days = np.sum(np.maximum(daily_mean_temp - base_temp, 0), **kwargs)
    return cooling_degree_days


def count_warm_days(tmax, **kwargs):
    """
    Calculate the Warm Days (TX90p).
    It's the percentage of days when daily maximum temperature is above the 90th percentile.
    """
    threshold = np.percentile(tmax, 90)
    warm_days = np.sum(tmax > threshold, **kwargs)
    return warm_days / len(tmax) * 100


def TX10p(tmax, **kwargs):
    """
    Calculate the Warm Days (TX10p).
    It's the percentage of days when daily maximum temperature is above the 90th percentile.
    """
    threshold = np.percentile(tmax, 10)
    warm_days = np.sum(tmax > threshold, **kwargs)
    return warm_days / len(tmax) * 100


def count_cold_nights(tmin, **kwargs):
    """
    Calculate the Cold Nights (TN10p).
    It's the percentage of days when daily minimum temperature is below the 10th percentile.
    """
    threshold = np.percentile(tmin, 10)
    cold_nights = np.sum(tmin < threshold, **kwargs)
    return cold_nights / len(tmin) * 100


def TN90p(tmin, **kwargs):
    """
    Calculate the Cold Nights (TN90p).
    It's the percentage of days when daily minimum temperature is below the 10th percentile.
    """
    threshold = np.percentile(tmin, 90)
    cold_nights = np.sum(tmin < threshold, **kwargs)
    return cold_nights / len(tmin) * 100

def calculate_TMm(data, **kwargs):
    """
    Calculate the Mean Daily Mean Temperature (TMm).
    It's the mean of the daily mean temperatures over a period.
    """
    mean_TM = np.mean(data, **kwargs)
    print(mean_TM)
    return mean_TM


def calculate_TXm(data, **kwargs):
    """
    Calculate the Mean TX (TXm).
    It's the mean of the daily maximum temperature.
    """
    mean_TX = np.mean(data, **kwargs)
    return mean_TX


def calculate_TNm(data, **kwargs):
    """
    Calculate the mean daily minimum temperature (TNm).
    """
    mean_TNm = np.mean(data, **kwargs)
    return mean_TNm


def calculate_gdd(data, base_temp=10, **kwargs):
    """
    Calculate Growing Degree Days (GDD).
    It's the sum of the differences between the daily mean temperature and the base temperature (10°C) for all days when the daily mean temperature is above 10°C.
    """
    daily_mean_temp = data
    growing_degree_days = np.sum(np.maximum(daily_mean_temp - base_temp, 0), **kwargs)
    return growing_degree_days


def calculate_extreme_temperature_range(data, **kwargs):
    # Calculate the maximum daily maximum temperature and minimum daily minimum temperature
    max_tx = np.sum(data, **kwargs)
    min_tn = np.sum(data, **kwargs)
    # Calculate Extreme Temperature Range (ETR)
    etr = max_tx - min_tn

    return etr


def calculate_daily_temperature_range(data, axis=0):
    # Calculate the daily max and min along the time axis
    max_tx = np.max(data, axis=axis)
    min_tn = np.min(data, axis=axis)

    # Calculate the daily temperature range for each day
    data_range = (max_tx - min_tn) / data.shape[axis]

    return data_range


def calculate_WSDI(TX_daily, base_period, percentile_90th):
    warm_spell_days = []
    consecutive_count = 0

    for temp in TX_daily:
        if temp > percentile_90th:
            consecutive_count += 1
        else:
            if consecutive_count >= 6:
                warm_spell_days.extend([1] * consecutive_count)
            consecutive_count = 0

    # Check if the last segment was a warm spell
    if consecutive_count >= 6:
        warm_spell_days.extend([1] * consecutive_count)

    # Step 3: Calculate the Warm Spell Duration Index (WSDI)
    annual_WSDI = sum(warm_spell_days)

    return np.array([annual_WSDI])  # Ensure to return an array


def calculate_90th_percentile(TX_daily, base_period=(1961, 1990), axis=None):
    TXin90 = np.percentile(TX_daily, 90)

    if axis is None:
        return calculate_WSDI(TX_daily, base_period, TXin90)
    elif axis == 0:
        return np.apply_along_axis(calculate_WSDI, 0, TX_daily, base_period=base_period, percentile_90th=TXin90)
    elif axis == 1:
        return np.apply_along_axis(calculate_WSDI, 1, TX_daily, base_period=base_period, percentile_90th=TXin90)
    else:
        raise ValueError("Invalid axis. Axis must be None, 0, or 1.")


def count_tm_ge_5(data, **kwargs):
    return np.mean(data >= 5, **kwargs)


def count_TM_lt_5(data, **kwargs):
    return np.mean(data < 5, **kwargs)


def count_tm_ge_10(data, **kwargs):
    return np.mean(data >= 10, **kwargs)


def count_TM_lt_10(data, **kwargs):
    return np.mean(data < 10, **kwargs)


def count_tx_ge_30(data, **kwargs):
    # Count the number of days where TX ≥ threshold
    days_tx_ge_30 = np.sum(data >= 30, **kwargs)

    return days_tx_ge_30


def count_tx_ge_35(data, **kwargs):
    # Count the number of days where TX ≥ threshold
    days_tx_ge_35 = np.sum(data >= 35, **kwargs)

    return days_tx_ge_35


def percentage_days_gt_50p(data, axis=None):
    # Calculate the percentile value
    percentile_value = np.percentile(data, 50, axis=axis)

    # Count the number of days where TX > 50th percentile
    days_gt_50p = np.sum(data > percentile_value, axis=axis)

    # Calculate the percentage
    total_days = np.prod(data.shape) if axis is None else data.shape[axis]
    percentage = (days_gt_50p / total_days) * 100

    return percentage


def count_tn_lt_2(data, **kwargs):
    # Count the number of days where TN < threshold
    days_tn_lt_2 = np.sum(data < 2, **kwargs)

    return days_tn_lt_2


def count_tn_lt_minus_2(data, **kwargs):
    days_tn_lt_minus_2 = np.sum(data < -2, **kwargs)

    return days_tn_lt_minus_2


def count_tn_lt_minus_20(data, **kwargs):
    days_tn_lt_minus_20 = np.sum(data < -20, **kwargs)
    return days_tn_lt_minus_20


def TXbdTNbd(data, d=None, **kwargs):
    """
    Calculates the annual count of consecutive d days where both TX < 5th percentile
    and TN < 5th percentile, where 10 >= d >= 2.

    Parameters:
    - data: numpy array or list of temperature data (TX and TN) for the year
    - d: number of consecutive days to consider (default: None)

    Returns:
    - count: the annual count of consecutive d days meeting the condition
    """
    if d is None:
        d = 2 or 10
    elif d < 2 or d > 10:
        raise ValueError("Invalid value for 'd'. The value of d should be between 2 and 10 (inclusive).")

    # Convert data to a numpy array if it is a list
    data = np.array(data)

    # Calculate the 5th percentile for TX and TN
    tx_percentile = np.percentile(data[:, 0], 5)
    tn_percentile = np.percentile(data[:, 1], 5)

    # Find the consecutive d days where both TX and TN are below the 5th percentile
    consecutive_count = 0
    count = 0
    for i in range(len(data)):
        if np.all(np.less(data[i, 0], tx_percentile)) and np.all(np.less(data[i, 1], tn_percentile)):
            consecutive_count += 1
            if consecutive_count == d:
                count += 1
        else:
            consecutive_count = 0

    return count


def Rx1day(daily_precip, **kwargs):
    return np.max(daily_precip, **kwargs)


def Rx5day(daily_precip, window=5, **kwargs):
    # Ensure the input array is a numpy array
    daily_precip = np.asarray(daily_precip)
    # Check if the input array has the correct dimensions
    if daily_precip.ndim != 3:
        raise ValueError("Input data must be a 3D array with dimensions (time, lat, lon)")
    # Calculate the rolling sum of precipitation over a 5-day window
    rolling_precip_sum = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window), mode='valid'), axis=0,
                                             arr=daily_precip)
    # Find the maximum value of the rolling sum along the specified axis
    max_5day_precip = np.max(rolling_precip_sum, **kwargs)
    return max_5day_precip


def calculate_sdii(data, threshold=1.0, **kwargs):
    # Step 1: Identify wet days (days with precipitation >= threshold)
    wet_days = np.sum(data > threshold, **kwargs)

    # Step 2: Calculate the number of wet days
    W = len(wet_days)  # Number of wet days

    if W == 0:
        return 0  # No wet days, return 0

    # Step 3: Calculate SDII
    SDII = wet_days / W

    return SDII


def count_days_prcp_ge_10mm(data, axis=0, **kwargs):
    return np.sum(data >= 10, axis=axis)


def count_days_prcp_ge_20mm(data, axis=0, **kwargs):
    return np.sum(data >= 20, axis=axis)


def count_days_prcp_ge_nnmm(data, threshold, axis=0, **kwargs):
    return np.sum(data >= threshold, axis=axis)


def max_dry_spell_length(data, threshold=1.0, axis=None):
    if axis is None:
        # No specific axis provided, handle as 1D data
        return max_length_of_dry_spell(data, threshold)
    elif axis == 0:
        # Axis 0: calculate for each column (time series for multiple locations)
        return np.apply_along_axis(max_length_of_dry_spell, 0, data, threshold)
    elif axis == 1:
        # Axis 1: calculate for each row (e.g., different time series)
        return np.apply_along_axis(max_length_of_dry_spell, 1, data, threshold)
    else:
        raise ValueError("Invalid axis. Axis must be None, 0, or 1.")


def max_length_of_dry_spell(data, threshold=1, **kwargs):
    consecutive_days = 0
    max_consecutive_days = 0

    for value in data:
        if value < threshold:
            consecutive_days += 1
            max_consecutive_days = max(max_consecutive_days, consecutive_days)
        else:
            consecutive_days = 0

    return max_consecutive_days


def max_wet_spell_length(data, threshold=1.0, axis=None):
    if axis is None:
        # No specific axis provided, handle as 1D data
        return _max_consecutive_days_above_threshold(data, threshold)
    elif axis == 0:
        # Axis 0: calculate for each column (time series for multiple locations)
        return np.apply_along_axis(_max_consecutive_days_above_threshold, 0, data, threshold)
    elif axis == 1:
        # Axis 1: calculate for each row (e.g., different time series)
        return np.apply_along_axis(_max_consecutive_days_above_threshold, 1, data, threshold)
    else:
        raise ValueError("Invalid axis. Axis must be None, 0, or 1.")


def _max_consecutive_days_above_threshold(data, threshold):
    max_length = 0
    current_length = 0

    for value in data:
        if value >= threshold:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length


def annual_total_precipitation_when_rr_above_95th_percentile(data, threshold=1.0, axis=None):
    # Calculate the 95th percentile threshold for wet days in the reference period
    wet_days = data[data >= threshold]
    percentile_95th = np.percentile(wet_days, 95)

    if axis is None:
        # No specific axis provided, handle as 1D data
        return _total_precipitation_above_threshold(data, percentile_95th, threshold)
    elif axis == 0:
        # Axis 0: calculate for each column (time series for multiple locations)
        return np.apply_along_axis(_total_precipitation_above_threshold, 0, data, percentile_95th, threshold)
    elif axis == 1:
        # Axis 1: calculate for each row (e.g., different time series)
        return np.apply_along_axis(_total_precipitation_above_threshold, 1, data, percentile_95th, threshold)
    else:
        raise ValueError("Invalid axis. Axis must be None, 0, or 1.")


def _total_precipitation_above_threshold(data, percentile_95th, threshold):
    total_precipitation = 0

    for value in data:
        if value >= threshold and value > percentile_95th:
            total_precipitation += value

    return total_precipitation


def annual_total_precipitation_when_rr_above_99th_percentile(data, threshold=1.0, axis=None):
    # Calculate the 95th percentile threshold for wet days in the reference period
    wet_days = data[data >= threshold]
    percentile_99th = np.percentile(wet_days, 99)

    if axis is None:
        # No specific axis provided, handle as 1D data
        return _total_precipitation_above_threshold(data, percentile_99th, threshold)
    elif axis == 0:
        # Axis 0: calculate for each column (time series for multiple locations)
        return np.apply_along_axis(_total_precipitation_above_threshold, 0, data, percentile_99th, threshold)
    elif axis == 1:
        # Axis 1: calculate for each row (e.g., different time series)
        return np.apply_along_axis(_total_precipitation_above_threshold, 1, data, percentile_99th, threshold)
    else:
        raise ValueError("Invalid axis. Axis must be None, 0, or 1.")


def _total_precipitation_above_threshold(data, percentile_99th, threshold):
    total_precipitation = 0

    for value in data:
        if value >= threshold and value > percentile_99th:
            total_precipitation += value

    return total_precipitation


def annual_total_precipitation_on_wet_days(data, threshold=1.0, **kwargs):
    # Sum the daily precipitation amounts along the time axis
    total_precipitation = np.sum(data >= threshold, **kwargs)

    return total_precipitation


def contribution_from_very_wet_days_R95pTOT(data, threshold=1.0, axis=None):
    # Calculate R95p (annual total precipitation when RR above 95th percentile)
    R95p = annual_total_precipitation_when_rr_above_95th_percentile(data, threshold=threshold, axis=axis)

    # Calculate PRCPTOT (annual total precipitation on wet days)
    PRCPTOT = annual_total_precipitation_on_wet_days(data, threshold=threshold, axis=axis)

    # Calculate R95pTOT (% contribution to total precipitation from very wet days)
    R95pTOT = (R95p / PRCPTOT) * 100

    return R95pTOT


def contribution_from_very_wet_days_R99pTOT(data, threshold=1.0, axis=None):
    # Calculate R95p (annual total precipitation when RR above 95th percentile)
    R99p = annual_total_precipitation_when_rr_above_99th_percentile(data, threshold=threshold, axis=axis)

    # Calculate PRCPTOT (annual total precipitation on wet days)
    PRCPTOT = annual_total_precipitation_on_wet_days(data, threshold=threshold, axis=axis)

    # Calculate R95pTOT (% contribution to total precipitation from very wet days)
    R99pTOT = (R99p / PRCPTOT) * 100

    return R99pTOT


def calculate_DTR(data, **kwargs):
    # Check if TX and TN are present in the data
    if 'TX' in data:
        TX = data['TX']
    else:
        TX = np.random.random((121, 115, 229)) * 50  # Generate random data if not present

    if 'TN' in data:
        TN = data['TN']
    else:
        TN = np.random.random((121, 115, 229)) * 100  # Generate random data if not present

    # Ensure the shapes of TX and TN match
    if TX.shape != TN.shape:
        raise ValueError("Shapes of TX and TN must match.")

    # Calculate the Daily Temperature Range (DTR)
    DTR = np.mean(TX - TN, **kwargs)
    return DTR


def calculate_ETR(data, **kwargs):
    # Check if TX and TN are present in the data
    # Check if TX and TN are present in the data
    if 'TX' in data:
        TX = data['TX']
    else:
        TX = np.random.random((121, 115, 229))  # Generate random data if not present

    if 'TN' in data:
        TN = data['TN']
    else:
        TN = np.random.random((121, 115, 229))  # Generate random data if not present

    # Ensure the shapes of TX and TN match
    if TX.shape != TN.shape:
        raise ValueError("Shapes of TX and TN must match.")

    # Calculate the Extreme Temperature Range (ETR) for each month
    ETR = np.sum(TX - TN, **kwargs)  # Calculate maximum TX and minimum TN for each month

    return ETR
# Index function dispatcher
def index_function_by_name(index_name, data, threshold=None, base_period=None, **kwargs):
    # take min temperature
    if index_name == 'FD':
        return count_frost_days(data, **kwargs)
    elif index_name == 'TR':
        return count_tropical_nights(data, **kwargs)
    elif index_name == 'TNx':
        return max_daily_min_temp(data, **kwargs)
    elif index_name == 'TNn':
        return min_daily_min_temp(data, **kwargs)
    elif index_name == 'TN10p':
        return count_cold_nights(data, **kwargs)
    elif index_name == 'TN90p':
        return TN90p(data, **kwargs)
    elif index_name == 'TNm':
        return calculate_TNm(data, **kwargs)
    elif index_name == 'TNlt2':
        return count_tn_lt_2(data, **kwargs)
    elif index_name == 'TNltm2':
        return count_tn_lt_minus_2(data, **kwargs)
    elif index_name == 'TNltm20':
        return count_tn_lt_minus_20(data, **kwargs)

    # take max temperature
    elif index_name == 'SU':
        return count_summer_days(data, **kwargs)
    elif index_name == 'ID':
        return count_icing_days(data, **kwargs)
    elif index_name == 'TXx':
        return max_daily_max_temp(data, **kwargs)
    elif index_name == 'TXn':
        return min_daily_max_temp(data, **kwargs)
    elif index_name == 'TXm':
        return calculate_TXm(data, **kwargs)
    elif index_name == 'TX90p':
        return count_warm_days(data, **kwargs)
    elif index_name == 'TX10p':
        return TX10p(data, **kwargs)
    elif index_name == 'TXge30':
        return count_tx_ge_30(data, **kwargs)
    elif index_name == 'TXge35':
        return count_tx_ge_35(data, **kwargs)
    elif index_name == 'TXgt50p':
        return percentage_days_gt_50p(data, **kwargs)
    elif index_name == 'Rx1day':
        return Rx1day(data, **kwargs)
    elif index_name == 'Rx5day':
        return Rx5day(data, **kwargs)
    elif index_name == 'WSDI':
        return calculate_90th_percentile(data, base_period=base_period, **kwargs)


    # take mran  temperature
    elif index_name == 'GSL':
        return count_growing_season_length(data, **kwargs)
    elif index_name == 'HDD':
        return count_heating_degree_days(data, **kwargs)
    elif index_name == 'CDD':
        return count_cooling_degree_days(data, **kwargs)
    elif index_name == 'Gdd':
        return calculate_gdd(data, 10, **kwargs)
    elif index_name == 'TMm':
        return calculate_TMm(data, **kwargs)
    elif index_name == 'TMge5':
        return count_tm_ge_5(data, **kwargs)
    elif index_name == 'TMlt5':
        return count_TM_lt_5(data, **kwargs)
    elif index_name == 'TMge10':
        return count_tm_ge_10(data, **kwargs)
    elif index_name == 'TMlt10':
        return count_TM_lt_10(data, **kwargs)



    # take min,max precipitation

    elif index_name == 'ETR':
        return calculate_ETR(data, **kwargs)
    elif index_name == 'DTR':
        return calculate_DTR(data, **kwargs)
    elif index_name == 'TXbdTNbd':
        return TXbdTNbd(data, threshold=threshold, **kwargs)



    # take the precipitation
    elif index_name == 'SDII':
        return calculate_sdii(data, threshold=threshold, **kwargs)
    elif index_name == 'R10mm':
        return count_days_prcp_ge_10mm(data, **kwargs)
    elif index_name == 'R20mm':
        return count_days_prcp_ge_20mm(data, **kwargs)
    elif index_name == 'Rnnmm':
        if threshold is None:
            raise ValueError("A threshold value must be provided for the 'Rnnmm' index.")
        return count_days_prcp_ge_nnmm(data, threshold, **kwargs)
    elif index_name == 'CDD':
        return max_dry_spell_length(data, **kwargs)
    elif index_name == 'CWD':
        return max_wet_spell_length(data, **kwargs)  # Add the new index condition here
    elif index_name == 'R95p':
        return annual_total_precipitation_when_rr_above_95th_percentile(data, threshold=threshold, **kwargs)
    elif index_name == 'R99p':
        return annual_total_precipitation_when_rr_above_99th_percentile(data, threshold=threshold, **kwargs)
    elif index_name == 'PRCPTOT':
        return annual_total_precipitation_on_wet_days(data, threshold=threshold, **kwargs)
    elif index_name == 'R95pTOT':
        return contribution_from_very_wet_days_R95pTOT(data, threshold=threshold, **kwargs)
    elif index_name == 'R99pTOT':
        return contribution_from_very_wet_days_R99pTOT(data, threshold=threshold, **kwargs)
    else:
        raise ValueError(f"Unknown index name: {index_name}")


def create_color_levels(aggregated_data, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    levels = np.linspace(aggregated_data.min(), aggregated_data.max(), num=cmap.N + 1)
    return levels, cmap


def create_polygon(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4):
    coords = [(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4), (lon1, lat1)]
    polygon = Polygon(coords)
    return polygon


def filter_data_by_season_and_year(data, time_variable, season, start_year, end_year):
    months = {
        'january': [1],
        'february': [2],
        'march': [3],
        'april': [4],
        'may': [5],
        'june': [6],
        'july': [7],
        'august': [8],
        'september': [9],
        'october': [10],
        'november': [11],
        'december': [12],
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11],
        'annual': list(range(1, 13))
    }

    selected_months = months.get(season.lower(), list(range(1, 13)))
    filtered_data = []
    filtered_dates = []

    for i, date in enumerate(
            cftime.num2date(time_variable[:], units=time_variable.units, calendar=time_variable.calendar)):
        if date.year in range(start_year, end_year + 1) and date.month in selected_months:
            # print(date)
            filtered_data.append(data[i])
            filtered_dates.append(date)
    filtered_data = np.array(filtered_data)
    # print(filtered_dates)
    return filtered_data, filtered_dates


def plot_time_custom_function_with_dates(file_path, variable_name, start_date=None, end_date=None,
                                         start_year=None, end_year=None, season='annual',
                                         index_name='TXx', threshold=20, base_period=None,
                                         data_type='temp', lon1=None, lat1=None, lon3=None, lat3=None):
    dataset = nc.Dataset(file_path)
    try:
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]
        var_name = None
        for varname in dataset.variables.keys():
            if variable_name in varname.lower():
                var_name = varname
                break
        if var_name is None:
            raise ValueError(f"Variable '{variable_name}' not found in the NetCDF file.")
        data = dataset.variables[var_name][:].squeeze()
        time_variable = dataset.variables['time']

        # Handle NaN and Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle different dimensions
        if len(data.shape) == 2:
            aggregated_data = data
        elif len(data.shape) >= 3:
            if start_date and end_date:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                start_idx = nc.date2index(start_date_obj, time_variable, select='nearest')
                end_idx = nc.date2index(end_date_obj, time_variable, select='nearest')
                sliced_data = data[start_idx:end_idx + 1]
            elif start_year and end_year and season:
                filtered_data, filtered_dates = filter_data_by_season_and_year(data, time_variable, season, start_year,
                                                                               end_year)
                sliced_data = filtered_data
            else:
                sliced_data = data

            # Aggregate data along the time axis (assuming the time axis is the first dimension)
            if len(sliced_data.shape) == 3:
                aggregated_data = index_function_by_name(index_name, sliced_data, axis=0, threshold=threshold, base_period=base_period)
            else:
                aggregated_data = index_function_by_name(index_name, sliced_data, axis=0, threshold=threshold, base_period=base_period)
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D+ data.")

        # Handle NaN and Inf values
        if np.any(np.isnan(aggregated_data)) or np.any(np.isinf(aggregated_data)):
            aggregated_data = np.nan_to_num(aggregated_data, nan=0.0, posinf=0.0, neginf=0.0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cmap_name = get_color_mapping_according_to_type(data_type)
        levels, cmap = create_color_levels(aggregated_data, cmap_name)
        # Ensure levels is sorted
        if levels is not None:
            levels = sorted(levels)
            # Contour levels must be increasing
            if levels[0] > levels[-1]:
                levels = levels[::-1]
            elif levels[0] == levels[-1]:
                levels = np.linspace(levels[0], levels[0] + 1, num=10)
            # print(f"Sorted levels: {levels}")
        else:
            levels = np.linspace(np.min(aggregated_data), np.max(aggregated_data), num=10)
            # print(f"Default levels: {levels}")
        contour = plt.contourf(xlon, xlat, aggregated_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
        ax.coastlines()
        cbar = plt.colorbar(contour, orientation='horizontal')
        num_ticks = 10
        ticks = np.linspace(aggregated_data.min(), aggregated_data.max(), num=num_ticks)
        cbar.set_ticks(ticks)
        cbar.set_label(var_name)
        if start_date and end_date:
            plt.title(f"{variable_name} from {start_date} to {end_date}")
        elif start_year and end_year and season:
            plt.title(f"{variable_name} from {start_year} to {end_year} ({season.capitalize()})")
        else:
            plt.title(f"{variable_name}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        if any(coord is None for coord in [lon1, lat1, lon3, lat3]):
            extent = [xlon.min(), xlon.max(), xlat.min(), xlat.max()]
        else:
            polygon = create_polygon(lon1, lat1, lon3, lat1, lon3, lat3, lon1, lat3)
            ax.add_geometries([polygon], ccrs.PlateCarree(), facecolor='none', alpha=0.3)
            x_min = min(lon1, lon3) - 1
            x_max = max(lon1, lon3) + 1
            y_min = min(lat1, lat3) - 1
            y_max = max(lat1, lat3) + 1
            extent = [x_min, x_max, y_min, y_max]

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Save the plot to a BytesIO object and encode it as a base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    finally:
        dataset.close()


@app.route('/plot_local', methods=['POST'])
def plot_endpoint():
    request_data = request.get_json()

    file_path = request_data.get('file_path')
    variable_name = request_data.get('var_name')
    start_date = request_data.get('start_date')
    end_date = request_data.get('end_date')
    start_year = request_data.get('start_year')
    end_year = request_data.get('end_year')
    season = request_data.get('season', 'annual')
    index_name = request_data.get('index_name', 'TXx')
    data_type = request_data.get('data_type', 'temp')
    lon1 = request_data.get('lon1')
    lat1 = request_data.get('lat1')
    lon3 = request_data.get('lon3')
    lat3 = request_data.get('lat3')

    try:
        img_str = plot_time_custom_function_with_dates(
            file_path, variable_name, start_date, end_date,
            start_year, end_year, season, index_name, data_type,
            lon1, lat1, lon3, lat3
        )
        return jsonify({"image": img_str}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_firebase', methods=['POST'])
def plot_firebase_endpoint():
    request_data = request.get_json()

    dataset_id = request_data.get('dataset_id')
    access = request_data.get('access')
    start_date = request_data.get('start_date')
    end_date = request_data.get('end_date')
    start_year = request_data.get('start_year')
    end_year = request_data.get('end_year')
    season = request_data.get('season', 'annual')
    index_name = request_data.get('index_name', 'TXx')
    lon1 = request_data.get('lon1')
    lat1 = request_data.get('lat1')
    lon3 = request_data.get('lon3')
    lat3 = request_data.get('lat3')

    ref = db.reference(f'datasets/{access}/{dataset_id}')
    dataset = ref.get()

    path_on_cloud = dataset['file_path_on_cloud']
    data_type = dataset['type']
    variable_name = dataset['var_name']

    file_name = f"{dataset['name']}.nc"
    bucket = storage.bucket()
    blob = bucket.blob(path_on_cloud)
    blob.download_to_filename(file_name)

    try:
        img_str = plot_time_custom_function_with_dates(
            file_name, variable_name, start_date, end_date,
            start_year, end_year, season, index_name, data_type,
            lon1, lat1, lon3, lat3
        )
        return jsonify({"image": img_str}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add_sector', methods=['POST'])
def add_sector_to_firebase():
    # sectors: { sector_name: { indexes: [index1, index2, ...], sector_name } }
    try:
        # Get data from the request
        data = request.get_json()
        # Extract parameters from the data
        sector_name = data.get('sector_name')
        indexes = data.get('indexes')

        sector_data = {
            'sector_name': sector_name,
            'indexes': indexes
        }
        ref = db.reference(f'sectors/{sector_name}')
        ref.set(sector_data)

        return jsonify({"message": "Sector added successfully"}), 200
    except FirebaseError as e:
        error_message = f'Error adding sector: {str(e)}'
        return jsonify({'error': error_message}), 500

@app.route('/add_index', methods=['POST'])
def add_index_to_firebase():
    # sectors: { sector_name: { indexes: [index1, index2, ...], sector_name } }
    # index: { index_name, index_description, sector_name, high_index_indication,
    # low_index_indication, has_base_period, has_threshold, base_period, threshold
    # moderate_range }
    try:
        # Get data from the request
        data = request.get_json()
        # Extract parameters from the data
        index_code = data.get('index_code')
        index_name = data.get('index_name')
        index_description = data.get('index_description')
        sector_name = data.get('sector_name')
        high_index_indication = data.get('high_index_indication')
        low_index_indication = data.get('low_index_indication')
        has_base_period = data.get('has_base_period')
        has_threshold = data.get('has_threshold')
        base_period = data.get('base_period')
        threshold = data.get('threshold')
        moderate_range = data.get('moderate_range')

        index_data = {
            'index_code': index_code,
            'index_name': index_name,
            'index_description': index_description,
            'sector_name': sector_name,
            'high_index_indication': high_index_indication,
            'low_index_indication': low_index_indication,
            'has_base_period': has_base_period,
            'has_threshold': has_threshold,
            'base_period': base_period,
            'threshold': threshold,
            'moderate_range': moderate_range,
        }
        ref = db.reference(f'sectors/{sector_name}/indexes/{index_code}')
        ref.set(index_data)

        return jsonify({"message": "Index added successfully"}), 200
    except FirebaseError as e:
        error_message = f'Error adding index: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/edit_index/<sector_name>/<index_code>', methods=['PUT'])
def edit_index_in_firebase(sector_name, index_code):
    try:
        data = request.get_json()
        ref = db.reference(f'sectors/{sector_name}/indexes/{index_code}')

        # Check if the index exists
        if not ref.get():
            return jsonify({"error": "Index not found"}), 404

        # Update index data
        ref.update(data)
        return jsonify({"message": "Index updated successfully"}), 200
    except FirebaseError as e:
        error_message = f'Error updating index: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/sectors', methods=['GET'])
def get_all_sectors():
    try:
        ref = db.reference('sectors')
        sectors = ref.get()
        if not sectors:
            return jsonify({"error": "No sectors found"}), 404
        return jsonify(sectors), 200
    except FirebaseError as e:
        error_message = f'Error retrieving sectors: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/indexes', methods=['GET'])
def get_all_indexes():
    try:
        ref = db.reference('sectors')
        sectors = ref.get()
        all_indexes = {}

        if not sectors:
            return jsonify({"error": "No indexes found"}), 404

        for sector_name, sector_data in sectors.items():
            indexes = sector_data.get('indexes', {})
            all_indexes[sector_name] = indexes

        return jsonify(all_indexes), 200
    except FirebaseError as e:
        error_message = f'Error retrieving indexes: {str(e)}'
        return jsonify({'error': error_message}), 500


if __name__ == "__main__":
    app.run(debug=True)



# in the beginning: we wanted the system to work on the data coming from the gov which is forcasted, thus the system was mainly done for
#                   Egypt gov to use, we wanted all the indexes funcs. to have all its parameters flixable for the admin to change for what
#                   he knows suits Egypt wither more than the europian difinition

# now: No data is coming from the gov, the system is expected to work similar in all condition with any data it's given
#      whatever the sources of this data, this change has affected the hoped outcome of the final product
#      because it makes no sense anymore for the admin to change the indexes equations parameters,
#      also losing the forcasted data advantage, for the code though we had to make it more flixable to handle any
#      NetCDF file thet it encounters due to the many sources of data expected,
#      handling NetCDF files leave us no choice but to use python for our logic,
#      having no exp what soever in using it as a backend we needed a place to store the data which lead us to Firebase
#      so the data is stored in Firebase, and the Logic is in python

# what we have accomplished so far: uploading datasets to Firebase from the Admin or the Analyst,
#                                   the Admin datasets are global while the ones from the analysts are for them only
#                                   the admin can do this too as he might want his uploaded dataset not to be global
#                                   the normal user can do the analysis on his own data too but it won't be stored
#                                   in the server (performing local analysis on the device)
#                                   the normal user can request to be an Analyst which the admin can approve or reject
#                                   we have finished implementing 47 indexes functions
#                                   and we have also implemented the result as a Time chart
#

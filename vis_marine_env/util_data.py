import os
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy.ndimage import zoom


def process_current_files(folder_path):
    nc_files = [f for f in os.listdir(folder_path) if f.startswith('BAL-NEMO_PHY-15minutes') and f.endswith('.nc')]
    nc_files.sort()


    for i in range(0, len(nc_files) - 1, 1):
        file1 = nc_files[i]
        file2 = nc_files[i + 1]

        time1 = file1.split('-')[-1].split('.')[0]
        time2 = file2.split('-')[-1].split('.')[0]

        if int(time2) - int(time1) != 12 and int(time2) - int(time1) != 88:
            print(f"The time index is not correct: {file1} and {file2}")
            continue

        with Dataset(os.path.join(folder_path, file1), 'r') as nc1, \
                Dataset(os.path.join(folder_path, file2), 'r') as nc2:

            # output as float32
            uo1 = nc1.variables['uo'][:].astype(np.float32)
            vo1 = nc1.variables['vo'][:].astype(np.float32)
            uo2 = nc2.variables['uo'][:].astype(np.float32)
            vo2 = nc2.variables['vo'][:].astype(np.float32)

            # process MaskedArray
            if isinstance(uo1, np.ma.MaskedArray):
                uo1 = uo1.filled(np.nan)
            if isinstance(vo1, np.ma.MaskedArray):
                vo1 = vo1.filled(np.nan)
            if isinstance(uo2, np.ma.MaskedArray):
                uo2 = uo2.filled(np.nan)
            if isinstance(vo2, np.ma.MaskedArray):
                vo2 = vo2.filled(np.nan)

        # merge
        combined_uo = np.concatenate((uo1, uo2), axis=0)
        combined_vo = np.concatenate((vo1, vo2), axis=0)
        combined_data = np.stack((combined_uo,combined_vo),axis=0)
        return combined_data
        # save as npy
        # output_path = os.path.join("/media/ubuntu/SOFT/weiyf/USV/Bal_cur", f"current_{time1}.npy")
        # np.save(output_path, combined_data)
        # print(f"Save in: {output_path} (Size: {combined_data.nbytes / 1024 / 1024:.2f}MB)")

def process_wav_files(folder_path):

    nc_files = [f for f in os.listdir(folder_path) if f.startswith('BAL-WAM_WAV') and f.endswith('.nc')]
    nc_files.sort()

    for i in range(0, len(nc_files) - 1, 1):
        file1 = nc_files[i]
        file2 = nc_files[i + 1]

        time1 = file1.split('WAV')[-1].split('.')[0]  #  2024010100
        time2 = file2.split('WAV')[-1].split('.')[0]  #  2024010112

        if int(time2) - int(time1) != 12 and int(time2) - int(time1) != 88:
            print(f"The time index not correct:: {file1} and {file2}")
            continue


        with Dataset(os.path.join(folder_path, file1), 'r') as nc1, \
                Dataset(os.path.join(folder_path, file2), 'r') as nc2:

            VHM1 = nc1.variables['VHM0'][:].astype(np.float32)
            VHM2 = nc2.variables['VHM0'][:].astype(np.float32)
            if isinstance(VHM1, np.ma.MaskedArray):
                VHM1 = VHM1.filled(np.nan)
            if isinstance(VHM2, np.ma.MaskedArray):
                VHM2 = VHM2.filled(np.nan)
        # merge
        combined_VHM = np.concatenate((VHM1, VHM2), axis=0)
        return combined_VHM
        # save as npy
        # np.save(output_path, combined_VHM)
        # print(f"Save in: {output_path}")

def process_and_merge_files(cur_data, wav_data, index=[0,0]):



        # process wave data
        # 1.  cut data
        cur_data = cur_data[:,:, index[0]:index[0]+120, index[1]:index[1]+210]
        wav_cropped = wav_data[:, index[0]:index[0]+120, index[1]+1:index[1]+211]  # (24, 120, 210)

        # 2.  (24→96)
        original_time = np.linspace(0, 1, 24)
        new_time = np.linspace(0, 1, 96)
        interpolated_wav = np.full((96, 120, 210), np.nan)

        for i in range(wav_cropped.shape[1]):
            for j in range(wav_cropped.shape[2]):
                ts = wav_cropped[:, i, j]
                if not np.isnan(ts).any():
                    f = interp1d(original_time, ts, kind='linear', bounds_error=False, fill_value="extrapolate")
                    interpolated_wav[:, i, j] = f(new_time)

        # cur_data (96,120,210) -> (1,96,120,210)
        interpolated_wav = interpolated_wav[np.newaxis, ...]

        #  (2,96,120,210) + (1,96,120,210) -> (3,96,120,210)
        merged_data = np.concatenate([cur_data, interpolated_wav], axis=0)
        return merged_data

def interpolate_array(array):
    target_shape = [3, 192, 120, 210]

    if array.shape[0] != target_shape[0] or array.shape[2] != target_shape[2] or array.shape[3] != target_shape[3]:
        raise ValueError("Dimension Error")

    zoom_factors = [
        1,
        target_shape[1] / array.shape[1],
        1,
        1
    ]


    interpolated = zoom(array, zoom_factors, order=1)

    return interpolated



# 使用示例
if __name__ == '__main__':
    folder_path = "./source"
    uv = process_current_files(folder_path)
    wav = process_wav_files(folder_path)
    print(uv.shape)
    print(wav.shape)
    processed_data = process_and_merge_files(uv,wav)
    print(processed_data.shape)
    interpolated_data = interpolate_array(processed_data)
    print(interpolated_data.shape)

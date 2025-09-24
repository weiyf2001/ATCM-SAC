from util_data import process_current_files
from util_data import process_wav_files
from util_data import process_and_merge_files
from util_data import interpolate_array
from ocean_field_visualization import VisOcean

if __name__ == '__main__':
    vis = VisOcean(img_saving_path='ocean_visualization.gif')
    folder_path = "./source"
    uv = process_current_files(folder_path)
    wav = process_wav_files(folder_path)
    index = [150,230] # Left bottom
    processed_data = process_and_merge_files(uv,wav,index)
    interpolated_data = interpolate_array(processed_data)

    for i in range(interpolated_data.shape[1]):

        vis.generate_img(interpolated_data[:,i,:,:])

    vis.save_gif()


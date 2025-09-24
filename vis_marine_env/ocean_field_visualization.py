from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import gc
import imageio
from PIL import Image
import numpy.ma as ma


class VisOcean:
    def __init__(self, img_saving_path=None):
        self.img_saving_path = img_saving_path
        self.loc_list_x = []
        self.loc_list_y = []
        self.frame_list = []

    def prep_arrow_plot(self, current_x, current_y, fig=plt):


        scale_factor = 1 / 15
        current_x = zoom(current_x, scale_factor, order=1)
        current_y = zoom(current_y, scale_factor, order=1)

        current_speed = np.sqrt(current_x ** 2 + current_y ** 2)
        mask = np.isnan(current_x) | np.isnan(current_y)
        current_x = np.where(mask, 0, current_x)
        current_y = np.where(mask, 0, current_y)
        current_speed = np.where(mask, np.nan, current_speed)

        nx, ny = current_x.shape[1], current_x.shape[0]
        x = np.linspace(0, 210, nx)
        y = np.linspace(0, 120, ny)
        X, Y = np.meshgrid(x, y)

        quiver = fig.quiver(
            X, Y,
            current_x, current_y,
            color='black',
            scale=5,
            width=0.002,
            angles='xy',
            alpha=0.8
        )

    def prep_contour_plot(self, field):


        a = np.linspace(0, 210, 210)
        b = np.linspace(0, 120, 120)
        x, y = np.meshgrid(a, b)
        wave_height = field
        land_mask = np.isnan(wave_height)
        masked_wave_height = ma.masked_array(wave_height, mask=land_mask)
        cmap_ocean = plt.cm.get_cmap('viridis')  # 或 'jet', 'plasma', 'YlGnBu' 等
        contour = plt.contourf(
            x, y, masked_wave_height,
            cmap=cmap_ocean,
            extend='both',
            zorder=1
        )

        plt.contourf(
            x, y, land_mask.astype(float),
            levels=[0.1, 1],
            colors=['#ddbc7d'],
            zorder=2
        )

        # 画海岸线
        plt.contour(
            x, y, land_mask.astype(float),
            levels=[0.1],
            colors=['black'],
            linewidths=0.8,
        )

        cbar = plt.colorbar(contour)
        cbar.set_label('Wave Height (m)')

    def generate_img(self, data):
        figure = plt.figure()

        gif_size=[16,9]
        current_x = data[0]
        current_y = data[1]
        wav = data[2]

        self.prep_contour_plot(wav)
        self.prep_arrow_plot(current_x,current_y)

        figure_temp = plt.gcf()
        figure_temp.set_size_inches(gif_size[0], gif_size[1])
        frame = figure_temp.canvas
        ag = frame.switch_backends(FigureCanvasAgg)
        ag.draw()
        A = np.asarray(ag.buffer_rgba())
        img = Image.fromarray(A)
        # plt.show()
        self.frame_list.append(img)
        plt.close()
        gc.collect()

    def save_gif(self, filename="ocean_animation.gif", duration=0.5):
        if len(self.frame_list) == 0:
            print("No frames to save!")
            return

        # for i, img in enumerate(self.frame_list):
        #     print(f"Frame {i}: hash={hash(img.tobytes()[:100])}")

        imageio.mimsave(filename, self.frame_list, duration=duration, loop=0)
        print(f"GIF saved as {filename}")
    def clear_frames(self):
        self.frame_list = []
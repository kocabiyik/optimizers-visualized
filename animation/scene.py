class Scene:
    def __init__(self, dir_to_save, frame_rate = 29):
        self.dir_to_save = dir_to_save
        self.frame_rate = frame_rate

    def printname(self):
        print(self.frame_rate, self.dir_to_save)

class Video(Scene):
    def __init__(self, video_name, dir_to_save, frame_rate, duration_in_sec=5, resolution = '1920x1080', plot_naming = 'plot_%03d', video_format='mp4'):
        super().__init__(dir_to_save, frame_rate)
        self.video_name = video_name+'.'+video_format
        self.duration_in_sec = duration_in_sec
        self.resolution = resolution
        self.plot_naming = plot_naming
        
    def __repr__(self):
        return f'Video({self.video_name!r}, {self.frame_rate!r} fps, with {self.resolution!r}p resolution)'

    def get_fmpeg_video_cmd(self):
        return 'ffmpeg  -framerate {frame_rate} -i {dir_name}/{plot_naming}.png -s:v {resolution} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -r 30  {video_name} -y' \
              .format(frame_rate=self.frame_rate,
                      dir_name = self.dir_to_save,
                      plot_naming = self.plot_naming,
                      resolution = self.resolution,
                      video_name = self.video_name
                         )

class GIFfromMP4Video(Scene):
    def __init__(self, file_name, dir_to_save, frame_rate):
        super().__init__(dir_to_save, frame_rate)
        self.video_file_name = file_name+'.mp4'
        self.gif_file_name = file_name+'.gif'
    
    def __repr__(self):
        return f'GIF({self.gif_file_name!r} from {self.video_file_name!r})'
    
    def get_fmpeg_gif_cmd(self):
        """Constructs and returns ffmpeg command"""
        return f'ffmpeg -i {self.video_file_name} -vf palettegen palette.png -y \n\
ffmpeg -i {self.video_file_name} -pix_fmt rgb24 -i palette.png -lavfi paletteuse {self.gif_file_name} \n\
rm palette.png -y'
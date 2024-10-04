""" Flip all videos horizontally so that they are all right-handed """

import os
from moviepy.editor import VideoFileClip, vfx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
input_folder = './dataset/08'
output_folder = './dataset/08_flipped'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.mp4')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        video = VideoFileClip(input_path)

        flipped_video = video.fx(vfx.mirror_x)

        flipped_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

        video.close()
        flipped_video.close()

print('Processing complete!')

import os
import tempfile
import shutil
import time
import base64
import glob
import mimetypes
import yaml
import imageio
import numpy as np
import subprocess
import argparse
import cv2
import torch
from os import path as osp
from skimage.transform import resize
from skimage import img_as_ubyte
import streamlit as st
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

#Synthetic Video Generation Module
def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(
            np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(
                out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_video_meta_info(video_path):
    ret = {}
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print(e.stderr)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret


def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            print(video_path)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)  # TODO support custom device
    else:
        face_enhancer = None

    reader = Reader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    while True:
        img = reader.get_frame()
        if img is None:
            break

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            writer.write_frame(output)

        torch.cuda.synchronize(device)
        pbar.update(1)

    reader.close()
    writer.close()

def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png')
        args.input = tmp_frames_folder

    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        inference_video(args, video_save_path)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', f'{args.output}/{args.video_name}_vidlist.txt', '-c',
        'copy', f'{video_save_path}'
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
    if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
    os.remove(f'{args.output}/{args.video_name}_vidlist.txt')

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def generate(page, sourceImage, drivingVideo):
    print(page)
    if(page == "VoxCeleb"):
        checkpointPath = "config/vox-cpk.pth.tar"
        configPath = "config/vox-256.yaml"
    elif(page == "Fashion"):
        checkpointPath = "config/fashion.pth.tar"
        configPath = "config/vox-256.yaml"
    elif(page == "Taichi"):
        checkpointPath = "config/taichi-cpk.pth.tar"
        configPath = "config/taichi-256.yaml"
    
    source_image = imageio.imread(sourceImage)  # source image
    reader = imageio.get_reader(drivingVideo)  # driving video
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3]
                     for frame in driving_video]
    generator, kp_detector = load_checkpoints(
        config_path=configPath, checkpoint_path=checkpointPath, cpu=False)

    predictions = make_animation(source_image, driving_video, generator,
                                 kp_detector, relative=True, adapt_movement_scale=True, cpu=False)
    with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as fp2:
        temp_filename = fp2.name
    imageio.mimsave(temp_filename, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    return fp2.name

def enhancementGAN(input,page):
    print(page)
    start = time.time()
    args = argparse.Namespace()
    args.input=input
    output = os.path.split(input)
    args.output=output[0]
    args.model_name='RealESRGAN_x4plus'
    args.denoise_strength=0.5
    args.suffix='out'
    args.outscale=2
    args.tile=0
    args.tile_pad=10
    args.pre_pad=0
    if page=="VoxCeleb":
        args.face_enhance=True
    else:
        args.face_enhance=False
    args.fp32=True
    args.fps=None
    args.ffmpeg_bin='ffmpeg'
    args.extract_frame_first=True
    args.num_process_per_gpu=1
    args.alpha_upsampler='realesrgan'
    args.ext='auto'

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args)

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)
    end = time.time()
    print("Total Time Taken:", end-start)
    enhancedOutput = os.path.splitext(input)[0] + "_out.mp4" 
    return enhancedOutput

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

st.set_page_config(page_title="Puppet Master", layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def videoandimage(selectedPage):
    col1, col2 = st.columns(2)
# Create a file uploader widget for images
    with col1:
        st.write('<p class="third-class">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(" ", type=["jpg","png","jpeg"])
    column1, column2 = st.columns(2)
# Display the image if an image file is uploaded
    if uploaded_file is not None:
        with column1:
            image = Image.open(uploaded_file)
            st.image(image, caption='', use_column_width=True)
    with col2:
# File uploader widget
        st.write('<p class="third-class">Upload Video</p>', unsafe_allow_html=True)
        video_file = st.file_uploader(" ", type=["mp4", "mov", "avi"])

# Preview the uploaded video
    if video_file is not None:
        with column2:
            video_bytes = video_file.read()
            st.video(video_bytes)
            with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as fp:
                fp.write(video_bytes)

# Define a checkbox and get its state
    checkbox_state = st.checkbox('High Resolution (Significantly Increases Time)')

# Use the checkbox state in your Streamlit app
    if checkbox_state:
        st.write('<p class="fourth-class">High Resolution</p>', unsafe_allow_html=True)
        checkbox = 1
    else:
        st.write('<p class="fourth-class">Default</p>', unsafe_allow_html=True)
        checkbox = 0

    if st.button("Generate"):
        with st.spinner('Processing...'):   
            if checkbox==1:
                generated_filepath = generate(selectedPage,uploaded_file,fp.name)
                generated_filepath = enhancementGAN(generated_filepath,selectedPage)
            else:
                generated_filepath = generate(selectedPage,uploaded_file,fp.name)
        st.success('Done!')

        # error 
        generated_video = open(generated_filepath, 'rb')
        generated_video = generated_video.read()
        st.markdown(
        f'<div style="display: flex; justify-content: center;"><video width="800" height="600" controls><source src="data:video/mp4;base64,{base64.b64encode(generated_video).decode()}" type="video/mp4"></video></div>',
        unsafe_allow_html=True,
        
        )



st.write('<p class="my-class">PUPPET MASTER</p>', unsafe_allow_html=True)
st.write('<p class="next-class">Generate synthetic videos</p>', unsafe_allow_html=True)

# Define a container with a border
container_style = 'display: flex; justify-content: center; align-items: center; text-align: center; padding-bottom:15px; padding-top:15px'

# Add an image inside the container
image_url = 'https://cdn-images-1.medium.com/v2/resize:fill:1600:480/gravity:fp:0.5:0.4/1*oUr68UTNN6H_iFw08J7O9Q.jpeg'
st.markdown('<div style="{}"><img src="{}" width="1366" height="414"></div>'.format(container_style, image_url), unsafe_allow_html=True)

# Create a sidebar with links to different pages
st.write('<p class="second-class">Modules</p>', unsafe_allow_html=True)


# Centered selection box with width 600px
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    page = st.selectbox(" ", ["VoxCeleb", "Fashion", "Taichi"], key="module-selector")
    st.markdown('<style>div[role="listbox"] ul {width: 600px !important; margin-left: auto; margin-right: auto;}</style>', unsafe_allow_html=True)

# Render the selected page
if page == "VoxCeleb":
    st.title("Voxceleb")
    videoandimage(page)

elif page == "Fashion":
    st.title("Fashion")
    videoandimage(page)

elif page == "Taichi":
    st.title("Taichi")
    videoandimage(page)
    

# add CSS style to center elements
st.write(
    f"""
    <style>
        .stButton button {{
            margin: 0 auto;
            display: block;
            
        }}
       
        
    </style>
    """,
    unsafe_allow_html=True
)
# css to style the text
st.markdown('''
    <style>
        .my-class {
            text-align: center;
            margin-top: 0;
            margin-bottom: 0;
            padding-top: 0;
            padding-bottom: 0;
            font-size: 50px;
            font-weight: bold;
            font-family: Bahnschrift;
        }

        .next-class {
            margin-top: 0;
            margin-bottom: 0;
            padding-top: 0;
            padding-bottom: 0;
            text-align: center;
            font-weight: normal;
            font-family: Bahnschrift;
            font-size: 20px;

        }
        .second-class {
            text-align: center;
            font-weight: bold;
            font-family: Bahnschrift;
            font-size: 42px;

        
        }
        .third-class {
            text-align: center;
            font-weight: normal;
            font-family: Bahnschrift;
            font-size: 24px;
            margin: 0rem;
            text-decoration: underline;

        
        }
        .fourth-class {
            text-align: center;
            font-weight: normal;
            font-family: Bahnschrift;
            font-size: 24px;
        }
    </style>
''', unsafe_allow_html=True)


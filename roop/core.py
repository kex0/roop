#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predicter import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, clean_temp_subpath, normalize_output_path

if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_paths')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_paths')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=suggest_frame_processors(), nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--keep-filenames', help='keep original filenames', dest='keep_filenames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_paths = args.source_paths
    roop.globals.target_paths = args.target_paths
    roop.globals.output_path = normalize_output_path(roop.globals.source_paths, roop.globals.target_paths, args.output_path)
    roop.globals.frame_processors = args.frame_processor
    roop.globals.headless = args.source_paths or args.target_paths or args.output_path
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_audio = args.keep_audio
    roop.globals.keep_frames = args.keep_frames
    roop.globals.keep_filenames = args.keep_filenames
    roop.globals.many_faces = args.many_faces
    roop.globals.video_encoder = args.video_encoder
    roop.globals.video_quality = args.video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_frame_processors() -> List[str]:
    return ['face_swapper', 'face_enhancer']


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)

def create_filename(source_index, target_index):
    source_number = str(source_index + 1).zfill(3)
    target_number = str(target_index + 1).zfill(3)
    if roop.globals.keep_filenames:
        file_name, _ = os.path.splitext(os.path.basename(roop.globals.target_path))
        output_name = file_name + "_roop"
    else:
        file_name, _ = os.path.splitext(os.path.basename(roop.globals.output_path))
        output_name = file_name
        roop.globals.output_path = os.path.dirname(roop.globals.output_path)

    if len(roop.globals.target_paths) > 1:
        output_name = f"{output_name}_T{target_number}"
    if len(roop.globals.source_paths) > 1:
        output_name = f"{output_name}_S{source_number}"

    if not roop.globals.file_override:
        file_number = str(1).zfill(3)
        if any(os.path.exists(os.path.join(roop.globals.output_path, output_name) + ext) for ext in [".png", ".mp4"]):
            while True:
                new_output_name = f"{output_name}_{file_number}"
                if not any(os.path.exists(os.path.join(roop.globals.output_path, new_output_name) + ext) for ext in [".png", ".mp4"]):
                    output_name = new_output_name
                    break
                file_number = int(file_number)
                file_number += 1
                file_number = str(file_number).zfill(3)
    
    if is_image(roop.globals.target_path):
        return output_name + ".png"
    elif is_video(roop.globals.target_path):
        return output_name + ".mp4"
        


def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return

    # make sure roop.globals.target_paths and roop.globals.source_paths are lists
    if isinstance(roop.globals.target_paths, str):
        roop.globals.target_paths = [roop.globals.target_paths]
    elif isinstance(roop.globals.source_paths, str):
        roop.globals.source_paths = [roop.globals.source_paths]

    for target_index, target_path in enumerate(roop.globals.target_paths):
        roop.globals.target_path = target_path
        # update target thumbnails and extract frames
        if has_image_extension(roop.globals.target_path):
            image = ui.render_image_preview(roop.globals.target_path, (200, 200))
            ui.target_label.configure(image=image)
        else:
            video_frame = ui.render_video_preview(roop.globals.target_path, (200, 200))
            ui.target_label.configure(image=video_frame)
            update_status('Creating temp resources...')
            create_temp(roop.globals.target_path)
            update_status('Extracting frames...')
            extract_frames(roop.globals.target_path)
            temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)

        for source_index, source_path in enumerate(roop.globals.source_paths):
            roop.globals.source_path = source_path
            # update source thumbnails
            image = ui.render_image_preview(roop.globals.source_path, (200, 200))
            ui.source_label.configure(image=image)
            output_path = roop.globals.output_path

            # create the output filename
            output_name = create_filename(source_index, target_index)

            # make sure roop.globals.output_path is just path without filename 
            # if not roop.globals.keep_filenames:
            #     roop.globals.output_path = os.path.dirname(roop.globals.output_path)

            roop.globals.output_path = os.path.join(roop.globals.output_path, output_name)

            # process image to image
            if has_image_extension(roop.globals.target_path):
                if predict_image(roop.globals.target_path):
                    destroy()
                for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                    update_status('Progressing...', frame_processor.NAME)
                    frame_processor.process_image(roop.globals.source_path, roop.globals.target_path, roop.globals.output_path)
                    frame_processor.post_process()
                    release_resources()
                if is_image(roop.globals.target_path):
                    update_status('Processing to image succeed!')
                else:
                    update_status('Processing to image failed!')
                roop.globals.output_path = output_path
                continue
            # process image to videos
            if predict_video(roop.globals.target_path):
                destroy()
            for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                update_status('Progressing...', frame_processor.NAME)
                frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
                frame_processor.post_process()
                release_resources()
            # handles fps
            if roop.globals.keep_fps:
                update_status('Detecting fps...')
                fps = detect_fps(roop.globals.target_path)
                update_status(f'Creating video with {fps} fps...')
                create_video(roop.globals.target_path, fps)
            else:
                update_status('Creating video with 30.0 fps...')
                create_video(roop.globals.target_path)
            # handle audio
            if roop.globals.keep_audio:
                if roop.globals.keep_fps:
                    update_status('Restoring audio...')
                else:
                    update_status('Restoring audio might cause issues as fps are not kept...')
                restore_audio(roop.globals.target_path, roop.globals.output_path)
            else:
                move_temp(roop.globals.target_path, roop.globals.output_path)
            # clean and validate
            if is_video(roop.globals.output_path):
                update_status('Processing to video succeed!')
            else:
                update_status('Processing to video failed!')
            clean_temp_subpath(roop.globals.target_path, roop.globals.source_path)
            roop.globals.output_path = output_path
        clean_temp(roop.globals.target_path)


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()

from typing import Any, List, Callable
import os
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
import roop.ui as ui
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
# from deepface import DeepFace
import face_recognition
# import numpy as np

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None

# def verify_face(specific_target, reference_face):
#     if DeepFace.verify(specific_target, reference_face, model_name="Facenet", detector_backend="retinaface")['verified']:
#         return True
#     else:
#         return False

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame_path: str, matches_dict: dict) -> Frame:
    if roop.globals.only_specific_face:
        match_index = None

        # get key of the frame that is closest to one of the frames we run the face recognition on
        if len(matches_dict) > 1:
            temp_frame_path_int = int(os.path.basename(os.path.splitext(temp_frame_path)[0]))
            matches_dict_ints = [int(os.path.basename(os.path.splitext(key)[0])) for key in matches_dict.keys()]
            closest_match_int = min(matches_dict_ints, key=lambda x: abs(x - temp_frame_path_int))
            closest_match_key = [key for key in matches_dict.keys() if int(os.path.basename(os.path.splitext(key)[0])) == closest_match_int][0]
        else:
            closest_match_key = list(matches_dict.keys())[0]

        temp_frame = cv2.imread(temp_frame_path)
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            faces_bbox_list = []
            for target_face in many_faces:
                faces_bbox_list.append(target_face['bbox'])
            roop.globals.faces_bbox_dict[temp_frame_path] = faces_bbox_list

            # faces_bbox_dict_copy = roop.globals.faces_bbox_dict.copy()
            # faces_bbox_dict_ints = [int(os.path.basename(os.path.splitext(key)[0])) for key in faces_bbox_dict_copy.keys()]
            # closest_temp_frame_path_int = min(faces_bbox_dict_ints, key=lambda x: abs(x - temp_frame_path_int))
            # closest_temp_frame_path_key = [key for key in faces_bbox_dict_copy.keys() if int(os.path.basename(os.path.splitext(key)[0])) == closest_temp_frame_path_int][0]

            for i, bbox in enumerate(roop.globals.faces_bbox_dict[temp_frame_path]):
                if matches_dict[closest_match_key] is not None:
                    if check_overlap(bbox, matches_dict[closest_match_key]):
                        match_index = i
                        break
                else:
                    # get key of closest key that has non None value
                    non_none_indices = [i for i, value in enumerate(matches_dict.values()) if value is not None]
                    if non_none_indices:
                        match_index = list(matches_dict.keys()).index(closest_match_key)
                        result_index = min(non_none_indices, key=lambda x: abs(x - match_index))
                    else:
                        result_index = None
                    if result_index is not None:
                        closest_match_key = list(matches_dict.keys())[result_index]

                    if check_overlap(bbox, matches_dict[closest_match_key]):
                        match_index = i
                        break
                    
            if match_index is not None:
                # adding face index of swapped face which will be later used for face enhancing
                roop.globals.faces_bbox_dict[temp_frame_path] = [roop.globals.faces_bbox_dict[temp_frame_path], match_index]
                temp_frame = swap_face(source_face, many_faces[match_index], temp_frame)
    elif roop.globals.many_faces:
        temp_frame = cv2.imread(temp_frame_path)
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_directory_path: str, temp_frame_paths: List[str], matches_dict: dict, update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    for temp_frame_path in temp_frame_paths:
        result = process_frame(source_face, temp_frame_path, matches_dict)
        temp_output = os.path.join(temp_directory_path, os.path.basename(temp_frame_path))
        cv2.imwrite(temp_output, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    matches_dict = {}
    if roop.globals.only_specific_face:
        roop.globals.faces_bbox_dict = {}
        matches_dict = process_recognition([target_path])
    source_face = get_one_face(cv2.imread(source_path))
    result = process_frame(source_face, target_path, matches_dict)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    matches_dict = {}
    if roop.globals.only_specific_face:
        roop.globals.faces_bbox_dict = {}
        matches_dict = process_recognition(temp_frame_paths)
    temp_directory_path = os.path.join(os.path.dirname(temp_frame_paths[0]), os.path.splitext(os.path.basename(roop.globals.source_path))[0])
    if not os.path.exists(temp_directory_path):
        os.makedirs(temp_directory_path)
    roop.processors.frame.core.process_video(source_path, temp_directory_path, temp_frame_paths, matches_dict, process_frames)


def check_overlap(bbox1, bbox2):
    if bbox1 is not None and bbox2 is not None:
        overlap_threshold = 0.2
        x1_1, y1_1, x2_1, y2_1 = map(int, bbox1)
        x1_2, y1_2, x2_2, y2_2 = map(int, bbox2)
        intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        if intersection_area / bbox1_area >= overlap_threshold:
            return True
        else:
            return False
    

def process_recognition(temp_frame_paths: List[str]) -> dict:
    matches_dict = {}
    reference_face = face_recognition.load_image_file(roop.globals.specific_target)
    reference_encoding = face_recognition.face_encodings(reference_face)[0]

    print('[ROOP.FACE-SWAPPER] Running face recognition...')
    if not roop.globals.headless:
        ui.update_status('Running face recognition...')
    
    # run face recognition only on every "step" frames because its very slow
    step = 30
    for i in range(0, len(temp_frame_paths), step):
        temp_frame_path = temp_frame_paths[i] if i+step < len(temp_frame_paths) else temp_frame_paths[-1]
        temp_frame = face_recognition.load_image_file(temp_frame_path)
        face_locations = face_recognition.face_locations(temp_frame)
        face_locations = sorted(face_locations, key=lambda x: x[1])
        face_encodings = face_recognition.face_encodings(temp_frame, face_locations)

        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces([reference_encoding], face_encoding)
            if True in matches:
                face_location = ([face_locations[i][3], face_locations[i][0], face_locations[i][1], face_locations[i][2]]) # (top, right, bottom, left) to (left, top, right, bottom)
                matches_dict[temp_frame_path] = face_location
                break
            else:
                matches_dict[temp_frame_path] = None


    return matches_dict
import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.predicter import predict_frame
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_SPECIFIC_TARGET = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
specific_target_label = None
target_label = None
status_label = None


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, specific_target_label, target_label, status_label, file_override_switch

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{roop.metadata.name} {roop.metadata.version}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.2, relheight=0.25)

    specific_target_label = ctk.CTkLabel(root, text=None)
    specific_target_label.place(relx=0.4, rely=0.1, relwidth=0.2, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.7, rely=0.1, relwidth=0.2, relheight=0.25)

    source_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path())
    source_button.place(relx=0.1, rely=0.4, relwidth=0.2, relheight=0.1)

    specific_target_button = ctk.CTkButton(root, text='Select face to replace', cursor='hand2', command=lambda: select_specific_target())
    specific_target_button.place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.1)

    target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: [select_target_path(), update_dropdown()])
    target_button.place(relx=0.7, rely=0.4, relwidth=0.2, relheight=0.1)

    # queue_label = ctk.CTkLabel(root, text="10 items in the queue", cursor='hand2')
    # queue_label.place(relx=0.6, rely=0.5, relwidth=0.3)

    keep_fps_value = ctk.BooleanVar(value=roop.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep fps', variable=keep_fps_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_fps', not roop.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.6)

    keep_frames_value = ctk.BooleanVar(value=roop.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep frames', variable=keep_frames_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.65)

    specific_face_value = ctk.BooleanVar(value=roop.globals.only_specific_face)
    specific_face_switch = ctk.CTkSwitch(root, text='Only specific face', variable=specific_face_value, cursor='hand2', command=lambda: setattr(roop.globals, 'only_specific_face', specific_face_value.get()))
    specific_face_switch.place(relx=0.39, rely=0.52)

    keep_audio_value = ctk.BooleanVar(value=roop.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(root, text='Keep audio', variable=keep_audio_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_audio', keep_audio_value.get()))
    keep_audio_switch.place(relx=0.4, rely=0.6)

    many_faces_value = ctk.BooleanVar(value=roop.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, cursor='hand2', command=lambda: setattr(roop.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=0.4, rely=0.65)

    keep_filenames_value = ctk.BooleanVar(value=roop.globals.keep_filenames)
    keep_filenames_switch = ctk.CTkSwitch(root, text='Keep filenames', variable=keep_filenames_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_filenames', keep_filenames_value.get()))
    keep_filenames_switch.place(relx=0.7, rely=0.6)

    file_override_value = ctk.BooleanVar(value=roop.globals.file_override)
    file_override_switch = ctk.CTkSwitch(root, text='Override files', variable=file_override_value, cursor='hand2', command=lambda: setattr(roop.globals, 'file_override', file_override_value.get()))
    file_override_switch.place(relx=0.7, rely=0.65)

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.15, rely=0.75, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(root, text='Destroy', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=0.4, rely=0.75, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=0.65, rely=0.75, relwidth=0.2, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Become a GitHub Sponsor', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('RoopDonate').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://github.com/sponsors/s0md3v'))

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider, selected_preview_dropdown

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True, ipady=5)

    selected_preview_values = roop.globals.target_paths
    selected_preview_variable = ctk.StringVar(value=roop.globals.selected_preview)
    selected_preview_dropdown = ctk.CTkComboBox(preview, values=selected_preview_values, variable=selected_preview_variable, cursor='hand2', command=lambda variable: [setattr(roop.globals, 'selected_preview', variable), init_preview(), update_preview()])
    selected_preview_dropdown.pack(side="left", ipadx=50)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def select_specific_target() -> None:
    global RECENT_DIRECTORY_SPECIFIC_TARGET

    PREVIEW.withdraw()
    specific_target = ctk.filedialog.askopenfilename(title='select source images', initialdir=RECENT_DIRECTORY_SPECIFIC_TARGET)
    if is_image(specific_target):
        roop.globals.specific_target = specific_target
        RECENT_DIRECTORY_SPECIFIC_TARGET = os.path.dirname(roop.globals.specific_target)
        image = render_image_preview(roop.globals.specific_target, (200, 200))
        specific_target_label.configure(image=image)
    else:
        roop.globals.specific_target = None
        specific_target_label.configure(image=None)


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE

    PREVIEW.withdraw()
    source_paths = ctk.filedialog.askopenfilenames(title='select source images', initialdir=RECENT_DIRECTORY_SOURCE)
    roop.globals.source_paths = source_paths
    source_path = source_paths[0]
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        image = render_image_preview(roop.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        roop.globals.source_path = None
        source_label.configure(image=None)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET

    PREVIEW.withdraw()
    target_paths = ctk.filedialog.askopenfilenames(title='select target images and/or videos', initialdir=RECENT_DIRECTORY_TARGET)
    roop.globals.target_paths = target_paths
    target_path = target_paths[0]
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = roop.globals.target_path if roop.globals.keep_filenames else os.path.dirname(roop.globals.target_path)
        image = render_image_preview(roop.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = roop.globals.target_path if roop.globals.keep_filenames else os.path.dirname(roop.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        roop.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT
    if roop.globals.keep_filenames:
        output_path = ctk.filedialog.askdirectory(title='save output', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = ctk.filedialog.asksaveasfilename(title='save output file', initialfile='output', initialdir=RECENT_DIRECTORY_OUTPUT)

    roop.globals.output_path = output_path
    RECENT_DIRECTORY_OUTPUT = roop.globals.output_path if roop.globals.keep_filenames else os.path.dirname(roop.globals.output_path)
    start()
        

def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif roop.globals.source_path and roop.globals.target_path:
        roop.globals.selected_preview = roop.globals.target_path
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(roop.globals.selected_preview):
        preview_slider.pack_forget()
    if is_video(roop.globals.selected_preview):
        video_frame_total = get_video_frame_total(roop.globals.selected_preview)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)

def update_dropdown():
    selected_preview_dropdown.configure(values=roop.globals.target_paths)
    selected_preview_dropdown.set(roop.globals.target_paths[0])

def update_preview(frame_number: int = 0) -> None:
    if roop.globals.source_path and roop.globals.target_path:
        temp_frame = get_video_frame(roop.globals.selected_preview, frame_number)
        if predict_frame(temp_frame):
            quit()
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(roop.globals.source_path)),
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

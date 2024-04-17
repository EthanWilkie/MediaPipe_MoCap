import multiprocessing
import time
import ffmpeg
import os
import mediapipe as mp
import cv2
import subprocess
import sys
import shutil
from functools import partial
from tkinter import Tk, filedialog
import pandas as pd
import csv


from landmark_config import LandmarkStyleConfig

# comment

res = "Result-Skeleton"

#Global Variable for Position Data
#LANDMARK_POSITION_DATA = []


def install(package):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}.")
        return False


class MediaPipeConfig:
    def __init__(
        self,
        line_thickness=2,
        landmark_styles={},
        excluded_connections=[],
        specific_connections=[],
    ):
        self.line_thickness = line_thickness
        self.landmark_styles = landmark_styles
        self.specific_connections = specific_connections
        self.excluded_connections = excluded_connections


class VideoProcessor:
    def __init__(self, video_path, mediapipe_config, frame_range):
        self.video_path = video_path
        self.frame_range = frame_range
        self.output_folder = f"./frames-{res}"
        self.cap = cv2.VideoCapture(self.video_path)
        self.pose = mp.solutions.pose.Pose()
        self.mediapipe_config = mediapipe_config
        self.landmark_positions = [] #addition via Ethan - list that stores landmark positions

    def resize_frame(self, frame, max_width=1080, max_height=1920):
        aspect_ratio = frame.shape[1] / frame.shape[0]
        if frame.shape[0] > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        return frame

    def process_frame(self, frame_counter, frame):
        frame = self.resize_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        print(".", end='', flush=True)

        
        
        landmark_positions_frame = []
        LANDMARK_DATA = []
        
    
        if results.pose_landmarks:
            # Extract landmark positions ############################### Ethan Addition
            LANDMARK_POSITION_DATA = []

            for landmark in mp.solutions.pose.PoseLandmark:
                landmark_id = landmark.value
                landmark_name = landmark.name
                landmark_x = results.pose_landmarks.landmark[landmark_id].x * frame.shape[1]  # Scale to frame width
                landmark_y = results.pose_landmarks.landmark[landmark_id].y * frame.shape[0]  # Scale to frame height
                landmark_positions_frame.append((landmark_x, landmark_y))
                LANDMARK_DATA.append((frame_counter, landmark_name, landmark_x, landmark_y))
                LANDMARK_POSITION_DATA.append((frame_counter, landmark_name, landmark_x, landmark_y))
                
                # Print landmark name along with its coordinates
                #landmark_name = landmark.name
                #print(f"Frame {frame_counter}, {landmark_name}: ({landmark_x}, {landmark_y})")
                # Write data to CSV
                #with open('landmark_data.csv', mode='a', newline='') as file:
                    #writer = csv.writer(file)
                    #writer.writerow([frame_counter, landmark_name, landmark_x, landmark_y])
                # Export data to Excel
             # Write landmark data to the CSV file
            with open('NEWlandmark_position_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write a row to denote the beginning of a new "sheet" for this frame
                #writer.writerow([f"--- Frame {frame_counter} ---"])
                # Write the landmark data for this frame
                writer.writerow(["Frame Number", "Landmark Name", "X Position", "Y Position"])
                for landmark_data in LANDMARK_POSITION_DATA:
                    writer.writerow(landmark_data)
                #writer.writerows(LANDMARK_POSITION_DATA)
                #for landmark_data in LANDMARK_POSITION_DATA:
                    #writer.writerow(landmark_data)
            
            #print(landmark_positions_frame)
            #print(LANDMARK_DATA)
            #print(LANDMARK_POSITION_DATA)
        


            ############################################################# Ethan Addition End

            mp_drawing = mp.solutions.drawing_utils
            frame_with_pose = frame.copy()

            # Create custom DrawingSpec for each landmark
            custom_landmark_style_dict = {
                i: mp_drawing.DrawingSpec(
                    color=self.mediapipe_config.landmark_styles[i]["color"],
                    thickness=self.mediapipe_config.landmark_styles[i]["thickness"],
                    circle_radius=self.mediapipe_config.landmark_styles[i][
                        "circle_radius"
                    ],
                )
                for i in self.mediapipe_config.landmark_styles
            }

            # Dynamically determine connections to include
            connections = [
                conn
                for conn in mp.solutions.pose.POSE_CONNECTIONS
                if self.mediapipe_config.landmark_styles[conn[0]]["thickness"]
                is not None
                and self.mediapipe_config.landmark_styles[conn[1]]["thickness"]
                is not None
            ]

            # Explicitly add specific connections
            connections.extend(self.mediapipe_config.specific_connections)

            # Draw landmarks and connections with custom styles
            mp_drawing.draw_landmarks(
                frame_with_pose,
                results.pose_landmarks,
                connections,  # Existing connections
                landmark_drawing_spec=custom_landmark_style_dict,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=self.mediapipe_config.line_thickness,
                ),
            )

            frame_filename = os.path.join(
                self.output_folder, f"frame_{frame_counter:04d}.jpg"
            )

            cv2.imwrite(frame_filename, frame_with_pose)
            # Write data to CSV
            #with open('landmark_data2.csv', mode='a', newline='') as file:
                #writer = csv.writer(file)
                #writer.writerow([frame_counter, landmark_name, landmark_x, landmark_y])


            

        else:
            frame_filename = os.path.join(
                self.output_folder, f"frame_{frame_counter:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)

    def process_video_segment(self):
        frame_counter = self.frame_range[0]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        while frame_counter <= self.frame_range[1]:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame_counter, frame)
            frame_counter += 1

        self.cap.release()
        cv2.destroyAllWindows()
        return self.output_folder


class VideoAssembler:
    def __init__(self, frame_folder):
        self.frame_folder = frame_folder

    def assemble_video(self):
        if not os.path.exists(self.frame_folder):
            print("Frame folder not found.")
            return

        frame_files = sorted(
            [f for f in os.listdir(self.frame_folder) if f.endswith(".jpg")]
        )
        if not frame_files:
            print("No frame files found in the folder.")
            return

        output_path = f"final_output-{os.path.basename(self.frame_folder)}-27.mp4"

        input_stream = ffmpeg.input(
            os.path.join(self.frame_folder, "frame_%04d.jpg"), framerate=30
        )

        num_threads = multiprocessing.cpu_count()  # Get the number of available threads

        print('\n Threads: '+str(num_threads))

        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            **{"codec:v": "libx264"},
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Pad to even dimensions
            preset="veryfast",
            crf=27,
            threads=num_threads,
        )

        # overwrite = ffmpeg.overwrite_output(output_stream)

        ffmpeg.run(output_stream, quiet=False, overwrite_output=True)

        # input_stream = input(input_path, framerate=self.frame_rate)
        # output_stream = output(
        #     input_stream, output_path, **{"codec:v": self.codec}, preset="fast", crf=24, threads=num_threads
        # )


# def get_user_selection_inquirer(prompt, options):
#     questions = [
#         inquirer.List(
#             "select",
#             message=prompt,
#             choices=options,
#         ),
#     ]
#     answers = inquirer.prompt(questions)
#     return answers["select"]

def replace_drawing_utils_if_needed():
    # Update with the correct path
    custom_file_path = "C:\\Users\\ethan\\OneDrive\\Desktop\\Parados As. 1\\drawing_utils\\custom_drawing_utils.py" 
    #'drawing_utils/custom_drawing_utils.py'
    mp_file_path = os.path.join(
        os.path.dirname(mp.__file__), 'python', 'solutions', 'drawing_utils.py')
    # Check if the first line of the existing file matches the specified comment
    try:
        with open(mp_file_path, 'r') as file:
            first_line = file.readline()
            if '# modified by Parados' not in first_line:
                shutil.copyfile(custom_file_path, mp_file_path)
                print("MediaPipe drawing_utils.py replaced with custom version.")
            else:
                print("Custom MediaPipe drawing_utils.py already in place.")
    except IOError as e:
        print(
            f"Error occurred while checking or replacing drawing_utils.py: {e}")


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def split_video(video_path, num_threads):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_thread = frame_count // num_threads
    frame_ranges = []

    for i in range(num_threads):
        start_frame = i * frames_per_thread
        end_frame = start_frame + frames_per_thread - 1
        frame_ranges.append((start_frame, end_frame))

    # Make sure the last thread processes any remaining frames
    frame_ranges[-1] = (frame_ranges[-1][0], frame_count - 1)

    return frame_ranges


def process_video_segment(args):
    video_path, frame_range, mediapipe_config = args
    video_processor = VideoProcessor(video_path, mediapipe_config, frame_range)
    video_processor.process_video_segment()


def main():
    # Define options
    view_options = ["Front", "Left", "Right", "Back"]
    category_options = [
        "Upper Body Exercises",
        "Lower Body Exercises",
        "Rehabilitation Exercises",
        "Functional Movements",
        "Shooting",
    ]

    # # Get user selection through interactive CLI
    # selected_view = get_user_selection_inquirer("Select View", view_options)
    # selected_category = get_user_selection_inquirer(
    #     "Select Exercise Category", category_options
    # )

    # Default Selection:
    selected_view = 'Front'
    selected_category = 'Upper Body Exercises'

    video_path = filedialog.askopenfilename(title="Select MP4 file", filetypes=[("MP4 files", "*.mp4")]) #'./videos/squat1080.mp4'

    output_folder = os.path.join(".", f"frames-{res}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f"Folder '{output_folder}' already exists, skipping creation.")

    # Get video resolution
    replace_drawing_utils_if_needed()

    video_width, video_height = get_video_resolution(video_path)
    scale = 1

    # Calculate the scale based on video resolution
    if video_width <= 1080:
        scale = video_width / 1080  # You can adjust this value as needed

    style_config = LandmarkStyleConfig(selected_view, selected_category, scale)

    mediapipe_config = MediaPipeConfig(
        line_thickness=style_config.global_line_thickness,
        landmark_styles=style_config.landmark_styles,
        excluded_connections=style_config.excluded_connections,
        specific_connections=style_config.specific_connections,
    )

    # Measure the time for VideoAssembler
    start_time_video_processor = time.time()

    num_threads = multiprocessing.cpu_count()

    # Split the video into segments
    frame_ranges = split_video(video_path, num_threads)

    # Create a list of arguments for each segment
    process_args = [(video_path, frame_range, mediapipe_config)
                    for frame_range in frame_ranges]

    # Initialize a multiprocessing pool
    with multiprocessing.Pool(len(frame_ranges)) as pool:
        # Process video segments in parallel
        pool.map(process_video_segment, process_args)

    # Wait for all processes to finish
    pool.close()
    pool.join()  # This line blocks until all processes are done

    end_time_video_processor = time.time()

    # After processing video segments, get the frame_folder path
    frame_folder = filedialog.askdirectory() #os.path.join(".", f"frames-{res}")

    # Measure the time for VideoAssembler
    start_time_video_assembler = time.time()
    video_assembler = VideoAssembler(frame_folder)
    video_assembler.assemble_video()
    end_time_video_assembler = time.time()

    # Calculate execution times
    execution_time_video_processor = end_time_video_processor - start_time_video_processor
    execution_time_video_assembler = end_time_video_assembler - start_time_video_assembler
    total_execution_time = end_time_video_assembler - start_time_video_processor

    # Display execution times
    print(
        f"Processing by MediaPipe took {execution_time_video_processor} seconds")
    print(
        f"FFMPEG assembled the frames into a final video in {execution_time_video_assembler} seconds")
    print(f"Total script execution time: {total_execution_time} seconds")
    
    
if __name__ == "__main__":
    # try:
    #     import inquirer
    # except ImportError:
    #     print("Inquirer not found. Installing inquirer...")
    #     if install("inquirer"):
    #         print("Restarting script to load the newly installed package...")
    #         os.execl(sys.executable, sys.executable, *sys.argv)
    #     else:
    #         sys.exit(
    #             "Could not install required packages. Please install them manually and rerun the script."
    #         )
    main()



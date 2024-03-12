## Pi-Car-Turbo Project

### Latest files: camera_front -> inside-out

- Adapter -> Modification of the old vehicle using the 3D printer
  
- camera_rear: - hough          -> hough algorithm (#outdated) <br>
               - machine-learn  -> machine learning algorithm (#outdated) <br>
               - record-video   -> file to record a video (#finished) <br>
               - samples        -> samples of the car <br>

- camera_front: - sliding-windows -> sliding windows (#outdated) <br>
                - inside-out      -> efficent technique (#in development) <br>

- camera_record: - video recording -> file to record a video (#finished) <br>

- presentation:  - Explanation of the Hough algorithm: "Explanation of the Hough algorithm" <br>
                 - Explanation of sliding windows: "Explanation of sliding windows" <br>

### Overview
The Pi-Car-Turbo project, initiated by FIAN23, aims to develop a self-driving car equipped with the capability to autonomously navigate a racetrack. Based on the Race against AI project, the vehicle symbolizes advanced automation, incorporating essential elements like the automobile, a Raspberry Pi, a camera, and an electric engine.


### Project Goals
The primary objective is to create a self-driving car that can independently recognize a racetrack and maneuver along it, adhering to a predefined racing line. The project's initial challenge involves mastering straight-line driving between two markers. Subsequent phases will focus on enabling the car to navigate corners and progressively increase its speed on the track.

### Components
Car: The vehicle is designed for autonomous driving.

Raspberry Pi: Central to the project, the Raspberry Pi serves as the brain of the self-driving car.

Camera: An essential component for visual perception and track recognition.

Electric Engine: Powers the car's movement.

### Project Dependencies
We used the following [tutorial](https://www.youtube.com/watch?v=LECg-Gv5xjo&list=PL_r4rS7sBXUJUBmoPra9vMKZ6clpg_tdO&index=7) as the foundation of our code to understand the line detection algorithm. The implementation was carried out by ourselves; the tutorial served solely as guidance.

The project utilizes the [PiRacer-Py](https://github.com/twyleg/piracer_py) library. To set up the required repositories, follow these installation steps:

`python3 -m venv venv`

`source venv/bin/activate`

`nano requirements.txt  # Add "piracer-pi" to the file`

`pip install -r requirements.txt`



## Running the Project

### 1. Transfer Python Scripts to Raspberry Pi:

Use the scp command to push the Python scripts to the Raspberry Pi.

`scp C:\[folder]\stream_html.py [user@ipaddress]/[folder]`


### 2.Activate Virtual Environment:

Navigate to the project folder and activate the virtual environment.

`cd [folder]`

`source venv/bin/activate`


### 3. Run Streaming File:

Execute the Python script for streaming.

`python stream_html.py`


### View Pi-Racer Stream:

Access the Pi-Racer-Stream on any device using the provided IP address.

`[ipaddress]:8443/index.html`

## Personal adjustments

### Color Camera

If you want to have a colored image in the camera streams, you need to make the following adjustments:

1. Add the following to the stream_html file to obtain a colored image from the camera

`#colored frames`

`from color_camera import Camera`

`camera = Camera()`

2. Additionally, you must edit the frame in the following way to obtain RGB colors instead of BGR colors (default):

`frame = cvtColor(frame, COLOR_RGB2BGR)`

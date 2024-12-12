# FLIR Camera Image Acquisition and Autofocus System

This repository contains code for acquiring images from a FLIR camera and performing autofocus calibration using PID control. It is built using Python, the PySpin library for FLIR camera control, and integrates image processing, PID control, and real-time updates for adjusting focus.

## Features

- **Camera Initialization & Configuration**: 
  - Initialize and configure the FLIR camera (e.g., exposure time, pixel format, binning).
  - Manual or automatic exposure control.

- **Autofocus Control**: 
  - Real-time autofocus using PID control.
  - Ability to calibrate the autofocus system using a point spread function (PSF).

- **Image Acquisition & Preview**: 
  - Capture images from the camera and display them in a Tkinter GUI.
  - Supports zoom functionality for image preview.

- **PID Control for Autofocus**: 
  - Uses PID control to adjust the focus based on the PSF center in the acquired image.
  - Real-time adjustments of focus based on the control signal.

- **Integration with DAQ**: 
  - Send control signals to an analog output (via DAQ) to adjust the system's focus.
  - PID control updates the DAC value in real-time.

## Requirements

- Python 3.x
- PySpin (FLIR camera SDK)
- NIDAQmx (National Instruments DAQ API)
- Tkinter (for GUI)
- numpy, Pillow (for image processing)
- nidaqmx (for controlling the DAQ)

## Installation

1. Install the required dependencies:

   ```bash
   pip install numpy pillow nidaqmx pyspin
   ```

2. Install the FLIR PySpin SDK. Follow the [official PySpin installation guide](https://www.flir.com/support-center/).

3. Install the National Instruments NIDAQmx driver for controlling DAQ hardware.

4. Clone the repository:

   ```bash
   git clone https://github.com/your-username/flir-camera-autofocus.git
   ```

## Usage

### Camera Setup

1. Initialize the camera and configure the settings such as exposure time, pixel format, and binning using the provided functions.
   
2. Start recording by pressing the **Start** button in the Tkinter GUI. This will begin capturing images in real-time and processing them for autofocus.

3. Calibrate the autofocus by pressing the **Calibrate** button, which uses a PSF to adjust the system's focus using PID control.

4. Set different zoom levels using the provided buttons (`Zoom 1`, `Zoom 4`, `Zoom 10`).

5. Adjust the exposure settings manually, or reset to automatic exposure if needed.

### Image Acquisition & Autofocus

1. The system continuously acquires images from the FLIR camera.
2. The PID controller is used to adjust the focus based on the PSF center in each frame.
3. The `calib` function allows the user to calibrate the autofocus by using the PID controller to adjust the lens focus.
4. The **Stop** button halts the recording and stops the camera acquisition.

### DAQ Control

- The system communicates with a National Instruments DAQ to control the focus via an analog output.
- The system sends a voltage to the DAQ based on the focus adjustments calculated by the PID controller.

### Preview & Zoom

- **Preview**: Displays the captured image in real-time in the Tkinter window.
- **Zoom**: Zoom functionality is available to zoom into the center of the image, and the zoom level can be adjusted dynamically using the GUI.

## Functions

- `set_exposure`: Sets the camera's exposure time.
- `set_format`: Sets the camera's pixel format.
- `set_binning`: Configures the binning ratio.
- `reset_exposure`: Resets the exposure settings to automatic.
- `update_preview`: Updates the image preview in the Tkinter GUI.
- `calib`: Calibrates the autofocus system using a PSF and PID control.
- `start_recording`: Starts image acquisition.
- `stop_recording`: Stops image acquisition.
- `preview`: Starts the image preview.
- `reset`: Resets the system, including the PID controller and camera settings.

## Notes

- The system requires a FLIR camera and a compatible National Instruments DAQ device for operation.
- Ensure that the camera is connected and configured correctly before running the script.


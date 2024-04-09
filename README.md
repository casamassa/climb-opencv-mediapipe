# Project Climb - Vision Computer and AI

## Introduction

---

The idea of this project is develop an app to use in climb statistics, initially like move counter and distance counter.

## Dependencies and Installation

---

To install the application, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

## Usage

---

To use the applicaton, follow these steps:

1. Ensure that you have installed the required dependencies.

2. App to count moves from a video, run:

   ```
   python move_counter_video.py
   ```

   Wait until the end of the video, or press the 'q' key to finish immediately. Then look the results in the terminal log.

3. App to calculate a distance climbed from a video, run:

   ```
   python climb_distance.py
   ```

   Wait until the end of the video, or press the 'q' key to finish immediately. Then look the results in the terminal log.

4. There are some drafts py files used to study specific parts. To execute the drafts, run:

   ```
   python name_of_draft_file.py
   ```

## Notes

---

I choose use PyVenv to create virtual envorioment for my Python project, to ensure to use the Python version 3.11

First, make sure install Python 3.11 in your computer.

Make sure create your virtual envionment:

To create the virtual environment, run:

```
python3.11 -m venv name_of_virtual_env
```

For example: python3.11 -m venv venv

To activate your new virtual environment, run:

```
.\name_of_virtual_env\Scripts\Activate.ps1
```

For example: .\venv\Scripts\Activate.ps1

To deactivate your virtual environment, just run:
deactivate

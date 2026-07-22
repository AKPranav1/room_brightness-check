Download the FaceLandmarker model into this folder before running the app:

  curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

(Windows PowerShell equivalent:
  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
)

Expected file size is a few MB. If curl/PowerShell returns a small HTML/error
file instead, your network is blocking storage.googleapis.com — download it
from another network/browser and copy it in.

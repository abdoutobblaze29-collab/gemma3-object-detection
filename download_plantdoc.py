from roboflow import Roboflow

# 🔑 replace with your actual API key
rf = Roboflow(api_key="YOUR_API_KEY")

# access workspace + project
project = rf.workspace("joseph-nelson").project("plantdoc")

# choose version (latest is usually fine)
dataset = project.version(4).download("yolov8")

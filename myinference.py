from torch.xpu import device

from ultralytics import YOLO
import os
from pathlib import Path
import datetime

model = YOLO(Path("./runs/train/exp2/weights/best.pt"),
             task = "obb",
             verbose=True

             )
folderName = "../../myTestData/images_667"

datet = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outputFolder = "./output/" + datet

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder, exist_ok=True)

def process(result, filePath):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=filePath)  # save to disk


for filename in os.listdir(folderName):

    print("folder :", folderName + '/' + filename)

    results = model(folderName + '/' + filename,
                    # device="cuda:0",
                        device="cpu"
                    )

    filepath = outputFolder + "/" + filename

    print("filepath: ",  filepath)

    process(results[0], filepath)
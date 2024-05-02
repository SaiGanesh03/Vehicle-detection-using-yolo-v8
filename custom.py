from ultralytics import YOLO
model = YOLO('best4.pt')

results = model(source =0,show=True,conf=0.5,save=True)
# Advanced Driver Assistance using DeepSORT

## To run DeepSort, run in command line:
```
python main.py
```

## Define custom video path: 
On Line 65 in main.py change to the path of test video
```
self.input_path = 'Video_Files/fullvid.mp4'
```

## To Train Deep Descriptor Network: 
- Download Market1501 Dataset
- Run preprocess.py to convert it to Pytorch accessible format
- Change Line 16 to the path of the dataset folder 
```
root = "./dataset/"
```

## Final Result:
Video Link: https://youtu.be/qWqGCyW5Qfo


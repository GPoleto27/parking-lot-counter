# Parking lot counter
Computer vision application developed as a final test for "image processing and pattern recognition" class at UTFPR

## Objective
To detect and count empty/occupied parking spaces in an aerial view of a parking lot.

## Tools used
- [Python3](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [Numpy](https://numpy.org/)

## Methodology
- Select each parking spot on the video with `cv2.SelectROIs()` function
- Save the selected ROIs to [rois.txt](/rois.txt)
- Read each individual frame from [parking-lot.mp4](/videos/parking-lot.mp4) using `cv2.VideoCapture()`
- Apply the folowing filters:
    - `cv2.cvtColor()` to convert to grayscale:

        ![Grayscale image](/gray.png)
    - `cv2.GaussianBlur()` to apply a blur to reduce image noise from camera, location and compression:

        ![Blurred image](/blur.png)
    
    - `cv2.adaptiveThreshold()` to binarize the image:

        ![Binary image](/thresh.png)

- Now each pixel is either 0 or 255. We can notice that each parking spot with a vehicle in it will have more white pixels than the ones that are free, which means that the higher the mean value of each ROI, the higher the probability of the spot.

- From that we can empirically set a threshold for the mean value of each ROI, in this use case, it was set to 20

- Now iterate through each ROI, calculate it's mean value and check either it's occupied (above the threshold) ou free (below the threshold)


## Results

From all that processing we can now show which parking spots are free and count them as below:

![Result video](/videos/output.mp4)

We can also plot a time series of how many free parking spots there was each frame:
![Time series](/output.png)

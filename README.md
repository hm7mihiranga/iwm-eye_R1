# Blood vessel detection in eye fundus

## Development
Please try to follow [PEP8](https://peps.python.org/pep-0008/) style guide for Python. You can use [autopep8](https://pypi.org/project/autopep8/) formatter and [flake8](https://pypi.org/project/flake8/) as linter for this purpose. Don't set any additional arguments for these tools. Also note that each vessel detection method has own directory.

## Method 1 - Image Processing
Directory `image_processing` contains python script that uses image processing approach. Basically we can distinguish three stages of processing.

### Pre-processing
- conversion to grayscale (green channel)
- histogram equalizaiton (`CLAHE`)
- denoising (`non-local means`)

### Vessel Detection
- `frangi` ridge operator

### Post-processing
- thresholding
- removing small elements (`connected components with stats`)
- removing border (based on black color in hsv)

## Method 2 - Machine Learning
Directory `machine_learning` contains jupyter notebook that uses machine learning approach. Basically we can distinguish five stages of processing.

### Pre-processing
- image resizing

### Feature extraction
- dividing image into small windows (`view_as_windows`)
- calculating RGB mean, RGB standard deviation and image moments for each window - x values
- deciding whether center of window contains vessel (255) or not (0) - y values

### Learning
- undersampling of background class
- splitting data (`train_test_split`)
- using random forest
- tuning hyper-parameters (`RandomizedSearchCV`)
- evaluating model accuracy

### Classification
- using prepared model to detect vessels in windows

### Post-processing
- building image from windows and predicted labels
- removing small elements (`connected components with stats`) to improve quality

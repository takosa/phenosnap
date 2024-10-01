import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    layout='wide'
)
def smoothing_image(bgr_image, method, size, sigma):
    if method == 'Bilateral':
        return cv2.bilateralFilter(bgr_image, size, sigma, sigma)
    elif method == 'Gaussian':
        return cv2.GaussianBlur(bgr_image, (size, size), sigma)
    elif method == 'Median':
        return cv2.medianBlur(bgr_image, size)
    elif method == 'None':
        return bgr_image
    else:
        raise ValueError('Invalid method')

def extract_channel(bgr_image, channel):
    if channel == 'Blue':
        return cv2.extractChannel(bgr_image, 0)
    elif channel == 'Green':
        return cv2.extractChannel(bgr_image, 1)
    elif channel == 'Red':
        return cv2.extractChannel(bgr_image, 2)
    elif channel == 'Hue':
        return cv2.extractChannel(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV), 0)
    elif channel == 'Saturation':
        return cv2.extractChannel(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV), 1)
    elif channel == 'Value':
        return cv2.extractChannel(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV), 2)
    elif channel == 'L*':
        return cv2.extractChannel(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab), 0)
    elif channel == 'a*':
        return cv2.extractChannel(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab), 1)
    elif channel == 'b*':
        return cv2.extractChannel(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab), 2)
    elif channel == 'b* - a*':
        _, a, b = cv2.split(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab))
        x = (b.astype(np.int32) - a.astype(np.int32) + 254) // 2
        return x.astype(np.uint8)
    else:
        raise ValueError('Invalide channel name')

def thresholding_image(gray_image, method, min_thresh, max_thresh):
    if method == "Target > Otsu's threshold":
        return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Target <= Otsu's threshold":
        return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "Manual MIN threshold < Target <= Manual MAX threshold":
        _, smaller_than_max = cv2.threshold(gray_image, max_thresh, 255, cv2.THRESH_BINARY_INV)
        _, greater_than_min = cv2.threshold(gray_image, min_thresh, 255, cv2.THRESH_BINARY)
        return None, cv2.bitwise_and(smaller_than_max, greater_than_min)
    else:
        raise ValueError('Invalid method')

def smoothing_morphology(binary_image, method, opening_size, opening_iterations, closing_size, closing_iterations):
    if method == 'Opening':
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size)), iterations=opening_iterations)
    elif method == 'Closing':
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size)), iterations=closing_iterations)
    elif method == 'Opening then Closing':
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size)), iterations=opening_iterations)
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size)), iterations=closing_iterations)
    elif method == 'Closing then Opening':
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size)), iterations=closing_iterations)
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size)), iterations=opening_iterations)
    elif method == 'None':
        return binary_image
    else:
        raise ValueError('Invalid method')

def filling_holes(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(binary_image, contours, 255)
    return binary_image

def filter_out_ovjects(binary_image, remove_overhangs, min_area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary_image[labels == i] = 0
        if remove_overhangs:
            if stats[i, cv2.CC_STAT_TOP] == 0:
                binary_image[labels == i] = 0
            elif stats[i, cv2.CC_STAT_LEFT] == 0:
                binary_image[labels == i] = 0
            elif stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_LEFT] == binary_image.shape[1]:
                binary_image[labels == i] = 0
            elif stats[i, cv2.CC_STAT_HEIGHT] + stats[i, cv2.CC_STAT_TOP] == binary_image.shape[0]:
                binary_image[labels == i] = 0
    return binary_image

def image_processing_pipeline(bgr_image, smoothing_method, smoothing_size,
                              smoothing_sigma, channel, thresholding_method,
                              min_thresh, max_thresh, morphology_method, 
                              opening_morphology_size, opening_morphology_iterations,
                              closing_morphology_size, closing_morphology_iterations,
                              fill_holes):
    smoothed_image = smoothing_image(bgr_image, smoothing_method, smoothing_size, smoothing_sigma)
    grayscaled_image = extract_channel(smoothed_image, channel)
    thresh, binary_image = thresholding_image(grayscaled_image, thresholding_method, min_thresh, max_thresh)
    if fill_holes:
        binary_image = filling_holes(binary_image)
    binary_image = smoothing_morphology(binary_image, morphology_method, 
                                        opening_morphology_size, opening_morphology_iterations,
                                        closing_morphology_size, closing_morphology_iterations)
    binary_image = filter_out_ovjects(binary_image, remove_overhangs, min_area)
    return thresh, smoothed_image, grayscaled_image, binary_image


################### streamlit  ####################

################### sidebar    ####################
uploaded_files = st.sidebar.file_uploader(
    'Choose image files',
    ['jpg', 'png',],
    accept_multiple_files=True
)
smoothing_method = st.sidebar.selectbox(
    'Smoothing method',
    ['Bilateral', 'Gaussian', 'Median', 'None',],
)
smoothing_size = st.sidebar.slider(
    'Smoothing size',
    min_value=3,
    max_value=15,
    value=3,
    step=2
)
smoothing_sigma = st.sidebar.slider(
    'Smoothing sigma',
    min_value=3,
    max_value=15,
    value=3,
    step=2
)
channel = st.sidebar.selectbox(
    'Channel',
    ['Blue', 'Green', 'Red', 'Hue', 'Saturation', 'Value', 'L*', 'a*', 'b*', 'b* - a*'],
)
thresholding_method = st.sidebar.selectbox(
    'Thresholding method',
    ["Target > Otsu's threshold", "Target <= Otsu's threshold", "Manual MIN threshold < Target <= Manual MAX threshold",]
)
threshold = st.sidebar.slider(
    'Threshold',
    min_value=0,
    max_value=255,
    value=(127,255),
)

fill_holes = st.sidebar.checkbox(
    'Fill holes',
    value=False
)

morphology_method = st.sidebar.selectbox(
    'Morphology method',
    ['Opening', 'Closing', 'Opening then Closing', 'Closing then Opening', 'None'],
)
opening_morphology_size = st.sidebar.slider(
    'Opeing Morphology size',
    min_value=3,
    max_value=15,
    value=3,
    step=2
)
opening_morphology_iterations = st.sidebar.slider(
    'Opeing Morphology iterations',
    min_value=1,
    max_value=10,
    value=1,
)
closing_morphology_size = st.sidebar.slider(
    'Closing Morphology size',
    min_value=3,
    max_value=15,
    value=3,
    step=2
)
Closing_morphology_iterations = st.sidebar.slider(
    'Closing Morphology iterations',
    min_value=1,
    max_value=10,
    value=1,
)
remove_overhangs = st.sidebar.checkbox(
    'Remove overhangs',
    value=False
)
min_area = st.sidebar.number_input(
    'Minimum area',
    min_value=0,
    max_value=None,
    value=1000,
)


################### main panel ####################
selected_image_name = st.selectbox('Select image', [f.name for f in uploaded_files])
if len(uploaded_files) > 0:
    selected_image = {f.name: f for f in uploaded_files}[selected_image_name]
    image_bytes = selected_image.read()

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    threshold_otsu, smoothed_image, grayscaled_image, binary_image = image_processing_pipeline(img, smoothing_method, smoothing_size,
                                    smoothing_sigma, channel, thresholding_method,
                                    threshold[0], threshold[1], morphology_method,
                                    opening_morphology_size, opening_morphology_iterations,
                                    closing_morphology_size, Closing_morphology_iterations, fill_holes)
    
    row1 = st.columns(3)
    with row1[0]:
        st.image(smoothed_image, channels='BGR')
    with row1[1]:
        st.image(grayscaled_image)
    with row1[2]:
        st.image(binary_image)
    row2 = st.columns(3)
    with row2[0]:
        counts = np.unique_counts(grayscaled_image)
        fig, ax = plt.subplots()

        y_range = (0, counts.counts.max())
        if thresholding_method == "Target > Otsu's threshold":
            x_range = (threshold_otsu, counts.values.max())
        elif thresholding_method == "Target <= Otsu's threshold":
            x_range = (counts.values.min(), threshold_otsu)
        elif thresholding_method == "Manual MIN threshold < Target <= Manual MAX threshold":
            x_range = threshold
        
        ax.vlines(x_range, y_range[0], y_range[1], color='red', linestyles='dashed')
        ax.fill_between(x_range, y_range[0], y_range[1], color='red', alpha=0.3)
        
        ax.bar(counts.values, counts.counts, width=1)
        st.pyplot(fig)
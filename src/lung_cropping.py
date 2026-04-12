import cv2
import numpy as np
from PIL import Image
class LungCropping:
    def __init__(self, margin_pct=0.05, blur_kernel=(15, 15)):
        self.margin_pct = margin_pct
        self.blur_kernel = blur_kernel

    def __call__(self, image):
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()
        blur = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((25, 25), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if not contours:
            return image

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        img_h, img_w = gray.shape
        margin_x = int(w * self.margin_pct)
        margin_y = int(h * self.margin_pct)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        cropped_np = img_np[y1:y2, x1:x2]

        return Image.fromarray(cropped_np)
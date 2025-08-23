import cv2
import numpy as np

def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv_to_pil(image):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw_boxes_cv(pil_image, entities):
    image = pil_to_cv(pil_image)

    for ent in entities:
        x0, y0, x1, y1 = map(int, ent["bbox"])
        label = ent.get("predicted_class", "Unknown")
        conf = ent.get("confidence", 0.0)
        text = f"{label}: {conf:.2f}"

        cv2.rectangle(image, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=2)

        text_pos = (x0, max(y0 - 10, 10))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(image, (text_pos[0], text_pos[1] - h), (text_pos[0] + w, text_pos[1]), (0, 0, 255), -1)
        cv2.putText(image, text, text_pos, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return cv_to_pil(image)

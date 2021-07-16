import cv2
import numpy as np


def cv_imread(path, flag=1):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if cv_img.ndim == 2 and flag == 1:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    return cv_img


def cv_letterbox(image, h, w):
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image = cv2.resize(image, (nw, nh))
    if image.ndim == 3:
        new_image = np.zeros([w, h, 3], dtype=np.uint8)
    else:
        new_image = np.zeros([w, h], dtype=np.uint8)
    # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    new_image[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2: (w - nw) // 2 + nw] = image
    return new_image


def get_batch(idx_list, image_path_list, mask_path_list, height, width):
    image_batch = []
    mask_batch = []
    for idx in idx_list:
        image = cv_imread(image_path_list[idx])
        image = cv_letterbox(image, height, width)
        image = np.float32(image) / 255 - 0.5

        mask = cv_imread(mask_path_list[idx], 0)
        mask = cv_letterbox(mask, height, width)
        mask = np.float32(mask) / 255 - 0.5
        mask = np.expand_dims(mask, axis=-1)
        image_batch.append(image)
        mask_batch.append(mask)
    return image_batch, mask_batch


def create_compare_image(image, mask, predict_mask):
    print(np.amax(mask), np.amax(predict_mask))
    image_copy = (image.copy() + 0.5)
    mask_copy = (mask.copy() + 0.5)
    predict_mask_copy = np.clip((predict_mask.copy() + 0.5), 0., 1.)

    mask_copy = cv2.cvtColor(mask_copy[..., 0], cv2.COLOR_GRAY2BGR)
    predict_mask_copy = cv2.cvtColor(predict_mask_copy[..., 0], cv2.COLOR_GRAY2BGR)
    out_image = np.hstack([image_copy, mask_copy, predict_mask_copy])
    out_image = np.uint8(out_image * 255)
    return out_image


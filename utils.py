"""Utilities for skin AI project"""
import os
import os.path as op
from tqdm import tqdm
import json
from glob import glob
from shutil import copyfile
from typing import Optional
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split

CATEGORIES = [
       {"supercategory": "monkey", "id": 1, "name": "monkey"},

]

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def read_annotation_file(path):
    """Read annotation file"""
    return json.load(open(path, "r"))


def read_annotation_shapes(path: str):
    """Read annotation shapes from a given JSON path"""
    return json.load(open(path, "r"))["shapes"]


def convert_points_to_polygon(points):
    """Convert points to polygon as list of tuples"""
    return [tuple(l) for l in points]


def split_dataset(df: pd.DataFrame):
    """Split a given dataframe into training, validation, and test set"""
    df_train, df_val = train_test_split(df, test_size=0.15, random_state=42)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42)
    return df_train, df_val, df_test


def create_df_from_dir(path_dir: str, output_size: bool = False):
    """
    Create training dataframe from directory
    """
    img_paths = glob(f"{path_dir}/*.jpg")
    annotation_paths = glob(f"{path_dir}/*.json")
    img_df = pd.DataFrame(img_paths, columns=["image_path"])
    annotation_df = pd.DataFrame(annotation_paths, columns=["annotation_path"])
    img_df["img_name"] = img_df.image_path.map(lambda x: op.basename(x).replace(".jpg", ""))
    annotation_df["img_name"] = annotation_df.annotation_path.map(
        lambda x: op.basename(x).replace(".json", "")
    )
    df = img_df.merge(annotation_df, on="img_name")

    if output_size:
        print(f"Number of image: {len(img_paths)}")
        print(f"Number of annotation JSON: {len(annotation_paths)}")
        print(f"Total number of : {len(df)}")
    return df


def create_df_from_dir_(img_dir: str, annoatation_dir: str, output_size: bool = False):
    """
    Create training dataframe from directory
    """
    img_paths = glob(f"{img_dir}/*.jpg")
    annotation_paths = glob(f"{annoatation_dir}/*.json")
    img_df = pd.DataFrame(img_paths, columns=["image_path"])
    annotation_df = pd.DataFrame(annotation_paths, columns=["annotation_path"])
    img_df["img_name"] = img_df.image_path.map(lambda x: op.basename(x).replace(".jpg", ""))
    annotation_df["img_name"] = annotation_df.annotation_path.map(
        lambda x: op.basename(x).replace("_c.json", "")
    )
    df = img_df.merge(annotation_df, on="img_name")

    if output_size:
        print(f"Number of image: {len(img_paths)}")
        print(f"Number of annotation JSON: {len(annotation_paths)}")
        print(f"Total number of : {len(df)}")
    return df


def create_dataset(df: pd.DataFrame, output_path: str):
    """
    Copy dataset to output path from a given dataframe.
    Dataframe should contain `image_path` and `annotation_path` in the columns.
    """
    if not op.exists(output_path):
        os.makedirs(output_path)
        print(f"Create {output_path} since it doesn't exist before")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_name = op.basename(row["image_path"])
        annotation_name = op.basename(row["annotation_path"])
        dest_image_path = op.join(output_path, image_name)
        dest_annotation_path = op.join(output_path, annotation_name)
        if not op.exists(dest_image_path):
            copyfile(row["image_path"], dest_image_path)
        if not op.exists(dest_annotation_path):
            copyfile(row["annotation_path"], dest_annotation_path)


def polygon_to_mask(image, polygon):
    """
    Convert polygon to mask from a given image
    """
    width, height = image.size
    mask = Image.new("1", (width, height), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask)
    return mask


def plot_poly(
    image_path: str,
    poly: list,
    is_resize: bool = True,
    image_size: Optional[tuple] = (1000, 1000),
    alpha: float = 0.5,
):
    """
    Plot polygon on top of image.

    image_path: str, path to image
    alpha: float, blending ratio

    Example
    =======
    >>> image = Image.open(row["image_path"])
    >>> poly = convert_points_to_polygon(read_annotation_file(row["annotation_path"])[0]["points"])
    >>> plot_poly(image, poly)
    """
    image = Image.open(image_path)
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    draw.polygon(poly, fill="red")
    image_blend = Image.blend(image, image2, alpha)

    if not is_resize:
        return image, image_blend

    image_blend_resize = ImageOps.contain(image_blend, image_size)
    image = ImageOps.contain(image, image_size)
    return image, image_blend_resize


def create_coco_data_dict(
    path: str,
    labels: list = ["melasma", "hori nevus", "solar lentigines"],
    start=0,
    categories=CATEGORIES,
):
    """
    Create COCO dataset to be saved in JSON format from a given path.
    Path should contain images and annotations.

    path: str, path to image and annotation JSON files
    labels: list, interested labels
    start: int, default 0, starting index of the image index
    categories: list, default CATEGORIES, COCO categories
    """
    # map between class name and id
    categories_dict = {d["name"].lower(): d["id"] for d in categories}

    df = create_df_from_dir(path)
    images, annotations = [], []
    for i, r in tqdm(df.iterrows(), total=len(df)):
        image_id = start + i
        image_path = r.image_path
        annotation_path = r.annotation_path
        image_name = op.basename(image_path)

        # Read annotation file
        raw_annotation_info = read_annotation_file(annotation_path)
        raw_annotations = raw_annotation_info["shapes"]

        # Calculate image size ratio from annotation file with input image
        original_image_width = raw_annotation_info["imageWidth"]
        original_image_height = raw_annotation_info["imageHeight"]

        raw_annotations = read_annotation_file(annotation_path)["shapes"]
        for annotation in raw_annotations:
            label = annotation["label"].lower()
            if label in labels:
                category_id = categories_dict[label]
                image = Image.open(image_path)
                image_width, image_height = image.size

                resize_raito = original_image_height / image_height

                if len(annotation.get("points")) is not None:
                    # points = annotation["points"]
                    # Calculate new position points
                    points = np.array(annotation["points"])
                    points = (points / resize_raito).round()

                    polygon = convert_points_to_polygon(points)

                    if len(polygon) <= 1:
                        continue

                    segmentation = np.hstack(polygon)
                    mask = polygon_to_mask(image, polygon)
                    masks = np.expand_dims(mask, -1)

                    # Create bbox bbox array [num_instances, (y1, x1, y2, x2)].
                    boxes = extract_bboxes(masks)
                    bbox = boxes[0]
                    width = bbox[3] - bbox[1]
                    height = bbox[2] - bbox[0]
                    coco_bbox = [int(bbox[1]), int(bbox[0]), int(width), int(height)]
                    area = float(width * height)
                    image_dict = {
                        "id": image_id,
                        "width": image_width,
                        "height": image_height,
                        "file_name": image_name,
                        "file_path": image_path,
                    }
                    annotation_dict = {
                        "id": image_id,
                        "image_id": image_id,
                        "label": label,
                        "category_id": category_id,
                        "segmentation": [segmentation.tolist()],
                        "bbox": coco_bbox,
                        "iscrowd": False,
                        "area": area,
                        "original_bbox": bbox.tolist(),
                        "points": points.tolist(),
                    }
                    images.append(image_dict)
                    annotations.append(annotation_dict)
    coco_data_dict = {
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    return coco_data_dict
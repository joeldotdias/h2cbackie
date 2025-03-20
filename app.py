import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import cv2

from scipy.spatial import cKDTree

app = Flask(__name__)
CORS(app)

TEMP_DIR = "uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


def xray_to_point_cloud_v2(img_path):
    # Load the image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement to better separate bone from tissue
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Use thresholding focused on bone density (bones appear brighter in X-rays/CT)
    # Higher threshold to isolate just the bones (adjust this value as needed)
    bone_threshold = 180  # Adjust this threshold to isolate bones
    _, bone_mask = cv2.threshold(enhanced, bone_threshold, 255, cv2.THRESH_BINARY)

    # Clean up the bone mask
    kernel = np.ones((3, 3), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel)

    # Remove small isolated pixels that might be noise
    contours, _ = cv2.findContours(
        bone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_contour_area = 50  # Adjust this value based on your image
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            cv2.drawContours(bone_mask, [contour], -1, 0, -1)

    # Remove bottom rows (if needed)
    bone_mask[bone_mask.shape[0] - 3 : bone_mask.shape[0], 0 : bone_mask.shape[1]] = 0

    # Find the bounding box of the bone pixels
    bone_pixels = np.where(bone_mask == 255)
    if len(bone_pixels[0]) == 0 or len(bone_pixels[1]) == 0:
        print("No bone pixels found. Try adjusting the bone threshold.")
        exit()

    xmin, ymin, xmax, ymax = (
        np.min(bone_pixels[1]),
        np.min(bone_pixels[0]),
        np.max(bone_pixels[1]),
        np.max(bone_pixels[0]),
    )
    print(xmin, xmax, ymin, ymax)

    # Crop the image and create a mask
    crop = img[ymin : ymax + 3, xmin:xmax]
    mask = bone_mask[ymin : ymax + 3, xmin:xmax] > 0

    # Convert to BGRA and apply transparency
    result = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (mask * 255).astype(np.uint8)

    # Save the intermediate transparent image
    output_path = "bone_only.png"
    cv2.imwrite(output_path, result)

    # Show both the original mask and the result for comparison
    comparison = np.hstack(
        [
            cv2.cvtColor(bone_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        ]
    )

    # Generate the point cloud from bone-only pixels
    bone_gray = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2GRAY)
    alpha = result[:, :, 3]
    h, w = bone_gray.shape

    # Create 3D points with better depth values for bones
    # Brighter pixels (bone) should have higher elevation
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()

    # Use exponential scaling to emphasize bone structure
    # Adjust these parameters to fine-tune the 3D appearance
    bone_emphasis = 2.0  # Higher means more pronounced bone elevation
    min_depth = 10.0
    max_depth = 300.0

    # Calculate Z with emphasis on bone density
    normalized_gray = bone_gray.flatten() / 255.0
    z = min_depth + (np.power(normalized_gray, bone_emphasis) * (max_depth - min_depth))

    # Only keep points that are part of the bone mask
    mask_points = alpha.flatten() > 200  # Higher threshold to ensure only bone
    x = x[mask_points]
    y = y[mask_points]
    z = z[mask_points]

    # Optional: Apply a median filter to smooth the bone surface
    if len(z) > 0:
        # Create a 2D grid for the points we have
        coords = np.column_stack((x, y))
        tree = cKDTree(coords)

        # For each point, find nearest neighbors and median filter the z values
        smoothed_z = np.zeros_like(z)
        for i in range(len(z)):
            neighbors = tree.query_ball_point(coords[i], r=3)  # 3-pixel radius
            if len(neighbors) > 1:
                smoothed_z[i] = np.median(z[neighbors])
            else:
                smoothed_z[i] = z[i]

        z = smoothed_z

    # Stack coordinates and save
    points = np.column_stack((x, y, z))
    np.savetxt(
        "bone_point_cloud.txt", points, fmt="%.6f", header="x y z", comments="# "
    )

    print(f"Bone-only image saved as {output_path}")
    print(f"Bone point cloud saved as bone_point_cloud.txt with {len(points)} points")
    return output_path


def xray_to_point_cloud_v1(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh[thresh.shape[0] - 3 : thresh.shape[0], 0 : thresh.shape[1]] = 0

    white = np.where(thresh == 255)
    xmin, ymin, xmax, ymax = (
        np.min(white[1]),
        np.min(white[0]),
        np.max(white[1]),
        np.max(white[0]),
    )
    print(xmin, xmax, ymin, ymax)

    # Crop the image and create a mask
    crop = img[ymin : ymax + 3, xmin:xmax]
    mask = thresh[ymin : ymax + 3, xmin:xmax] > 0

    # Convert to BGRA and apply transparency
    result = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (mask * 255).astype(np.uint8)

    # Save the intermediate transparent image
    output_path = "screw_bone.png"
    cv2.imwrite(output_path, result)

    gray_for_points = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2GRAY)
    alpha = result[:, :, 3]
    h, w = gray_for_points.shape

    # Create 3D points
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    z = (gray_for_points.flatten() / 255.0) * 250.0
    mask_points = alpha.flatten() > 0

    # Apply mask to keep only visible points
    x = x[mask_points]
    y = y[mask_points]
    z = z[mask_points]

    # Stack coordinates and save
    points = np.column_stack((x, y, z))
    point_cloud_path = os.path.join(TEMP_DIR, "pointcloud.txt")
    np.savetxt(
        point_cloud_path,
        points,
        fmt="%.6f",
        header="x y z",
        comments="# ",
    )
    return point_cloud_path


def txt_to_obj(inpath):
    outpath = os.path.join(TEMP_DIR, "pointcloud.obj")
    with open(inpath, "r") as infile, open(outpath, "w") as outfile:
        # i have no idea if this is necessary
        outfile.write("# OBJ file\n")

        next(infile)

        for line in infile:
            x, y, z = line.strip().split()
            outfile.write(f"v {x} {y} {z}\n")

    return outpath


@app.route("/upload", methods=["POST"])
def upload_xray():
    if "xray" not in request.files:
        return jsonify({"message": "Where's the fileee"}), 400

    xray_file = request.files["xray"]
    if not xray_file.filename:
        return jsonify({"message": "Where's the filenameeee"}), 400

    xray_path = os.path.join(TEMP_DIR, xray_file.filename)
    xray_file.save(xray_path)
    # insights = insights_from_xray(xray_path)
    point_cloud_path = xray_to_point_cloud_v1(xray_path)

    obj_out_path = txt_to_obj(point_cloud_path)

    os.remove(xray_path)
    os.remove(point_cloud_path)

    return send_file(obj_out_path, mimetype="application/octet-stream")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

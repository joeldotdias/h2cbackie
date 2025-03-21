import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

app = Flask(__name__)
CORS(app)

TEMP_DIR = "uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


def xray_to_volumetric_point_cloud(
    img_path,
    output_dir="./",
    bone_threshold=180,
    num_layers=10,
    max_depth=300.0,
    min_depth=10.0,
    smooth_iterations=2,
    noise_threshold=0.05,
):
    """
    Convert X-ray images to volumetric 3D point clouds highlighting bone structures.

    Parameters:
    ----------
    img_path : str
        Path to the input X-ray image
    output_dir : str
        Directory to save output files
    bone_threshold : int
        Threshold value to isolate bones (higher values isolate denser bones)
    num_layers : int
        Number of depth layers to create volume
    max_depth : float
        Maximum depth value for z-coordinate
    min_depth : float
        Minimum depth value for z-coordinate
    smooth_iterations : int
        Number of smoothing iterations to apply
    noise_threshold : float
        Threshold for noise removal (0-1, higher removes more points)

    Returns:
    -------
    str: Path to the generated point cloud file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement to better separate bone from tissue
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise before thresholding
    enhanced_smooth = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Use thresholding focused on bone density
    _, bone_mask = cv2.threshold(
        enhanced_smooth, bone_threshold, 255, cv2.THRESH_BINARY
    )

    # Clean up the bone mask with morphological operations
    kernel_close = np.ones((5, 5), np.uint8)  # Larger kernel for better connectivity
    kernel_open = np.ones((3, 3), np.uint8)

    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel_close)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel_open)

    # Remove small isolated areas (noise)
    contours, _ = cv2.findContours(
        bone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate average contour area to set dynamic threshold
    if len(contours) > 0:
        areas = [cv2.contourArea(contour) for contour in contours]
        avg_area = np.mean(areas)
        min_contour_area = max(
            50, avg_area * 0.05
        )  # Dynamic threshold based on average area
    else:
        min_contour_area = 50

    # Remove small contours (noise)
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            cv2.drawContours(bone_mask, [contour], -1, 0, -1)

    # Find the bounding box of the bone pixels
    bone_pixels = np.where(bone_mask == 255)
    if len(bone_pixels[0]) == 0 or len(bone_pixels[1]) == 0:
        raise ValueError("No bone pixels found. Try adjusting the bone threshold.")

    xmin, ymin, xmax, ymax = (
        np.min(bone_pixels[1]),
        np.min(bone_pixels[0]),
        np.max(bone_pixels[1]),
        np.max(bone_pixels[0]),
    )

    # Crop the image and create a mask
    crop = img[ymin : ymax + 1, xmin : xmax + 1]
    mask = bone_mask[ymin : ymax + 1, xmin : xmax + 1] > 0

    # Convert to BGRA and apply transparency
    result = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (mask * 255).astype(np.uint8)

    # Save the intermediate transparent image
    output_path = os.path.join(output_dir, "bone_only.png")
    cv2.imwrite(output_path, result)

    # Generate the point cloud from bone-only pixels
    bone_gray = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2GRAY)
    alpha = result[:, :, 3]
    h, w = bone_gray.shape

    # Create a distance transform to find the distance from the edge of the bone
    dist_transform = cv2.distanceTransform(
        (mask * 255).astype(np.uint8), cv2.DIST_L2, 5
    )

    # Normalize the distance transform
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # Apply Gaussian filter to smooth the distance transform
    dist_transform = gaussian_filter(dist_transform, sigma=2)

    # Create 3D points with better depth values for bones
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()

    # Use bone intensity for initial z-value
    normalized_gray = bone_gray.flatten() / 255.0

    # Combine bone intensity with distance transform for more volume
    dist_values = dist_transform.flatten()

    # Parameters for 3D volume control
    bone_emphasis = 2.2  # Higher means more pronounced bone elevation

    # Filter points that are part of the bone
    mask_points = alpha.flatten() > 200
    x_bone = x[mask_points]
    y_bone = y[mask_points]
    normalized_gray_bone = normalized_gray[mask_points]
    dist_values_bone = dist_values[mask_points]

    # Create multiple layers based on distance from edge and intensity
    points_list = []

    for layer in range(num_layers):
        # Layer factor decreases as we go deeper (1.0 to 0.1)
        layer_factor = 1.0 - (layer / num_layers) * 0.9

        # Use both distance from edge and intensity to determine z-value
        z = min_depth + (
            (
                np.power(normalized_gray_bone, bone_emphasis) * 0.7
                + np.power(dist_values_bone, 1.5) * 0.3
            )
            * (max_depth - min_depth)
            * layer_factor
        )

        # Create this layer's points
        layer_points = np.column_stack((x_bone, y_bone, z))
        points_list.append(layer_points)

    # Combine all layers
    points = np.vstack(points_list)

    # Add edge walls for more volume
    edge_mask = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    edge_points = np.where(edge_mask > 0)

    if len(edge_points[0]) > 0:
        edge_x = edge_points[1]
        edge_y = edge_points[0]

        # Sample edge points to reduce density (prevents too many points)
        sample_rate = max(1, len(edge_x) // 200)  # Sample to get around 200 edge points
        edge_x = edge_x[::sample_rate]
        edge_y = edge_y[::sample_rate]

        # For each sampled edge point, create vertical columns
        wall_points = []
        for i in range(len(edge_x)):
            depth_steps = 5
            for depth_step in range(depth_steps):
                depth = min_depth - (depth_step * max_depth / (2 * depth_steps))
                wall_points.append([edge_x[i], edge_y[i], depth])

        if wall_points:
            wall_points = np.array(wall_points)
            points = np.vstack((points, wall_points))

    # Advanced noise removal before smoothing
    # Calculate density to identify outliers
    if len(points) > 100:  # Only if we have enough points
        tree = cKDTree(points)
        # Count neighbors within radius
        density = np.array([len(tree.query_ball_point(point, r=5)) for point in points])
        density_threshold = np.percentile(density, noise_threshold * 100)

        # Keep points with sufficient neighbors (not isolated)
        mask_clean = density > density_threshold
        points = points[mask_clean]

    # Smooth the point cloud
    for _ in range(smooth_iterations):
        if len(points) > 10:  # Only apply smoothing if we have enough points
            # Build KD-tree for efficient nearest neighbor search
            tree = cKDTree(points[:, :2])

            # For each point, smooth z value based on neighbors
            smoothed_z = np.zeros(len(points))

            for i in range(len(points)):
                # Find neighbors within a radius
                neighbors = tree.query_ball_point(points[i, :2], r=3)
                if len(neighbors) > 1:
                    # Weighted average based on distance
                    neighbor_points = points[neighbors]
                    distances = np.linalg.norm(
                        neighbor_points[:, :2] - points[i, :2], axis=1
                    )
                    weights = 1.0 / (distances + 0.001)  # Avoid division by zero

                    # Exclude outlier z-values (more than 2 std from mean)
                    z_values = neighbor_points[:, 2]
                    z_mean = np.mean(z_values)
                    z_std = np.std(z_values)
                    if z_std > 0:
                        z_mask = np.abs(z_values - z_mean) < 2 * z_std
                        if np.sum(z_mask) > 0:  # Ensure we have points left
                            weights = weights * z_mask

                    if np.sum(weights) > 0:
                        smoothed_z[i] = np.sum(
                            weights * neighbor_points[:, 2]
                        ) / np.sum(weights)
                    else:
                        smoothed_z[i] = points[i, 2]
                else:
                    smoothed_z[i] = points[i, 2]

            points[:, 2] = smoothed_z

    # Add a fourth column for confidence (can be used for coloring in visualization)
    confidence = np.ones((len(points), 1))  # Default confidence of 1.0

    # Adjust confidence based on distance transform (higher in center of bone)
    for i in range(len(points)):
        x_idx, y_idx = int(points[i, 0]), int(points[i, 1])
        if 0 <= x_idx < w and 0 <= y_idx < h:
            confidence[i] = dist_transform[y_idx, x_idx] * 0.5 + 0.5  # Scale to 0.5-1.0

    # Add confidence as fourth column
    points_with_confidence = np.hstack((points, confidence))

    # Save the final point cloud
    output_filepath = os.path.join(output_dir, "volumetric_bone_point_cloud.txt")
    np.savetxt(
        output_filepath,
        points_with_confidence,
        fmt="%.6f",
        header="x y z confidence",
        comments="# ",
    )

    print(f"Bone-only image saved as {output_path}")
    print(
        f"Volumetric bone point cloud saved as {output_filepath} with {len(points)} points"
    )

    return output_filepath


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./"
        generate_volumetric_bone_point_cloud(img_path, output_dir)
    else:
        print("Usage: python script.py <image_path> [output_directory]")


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

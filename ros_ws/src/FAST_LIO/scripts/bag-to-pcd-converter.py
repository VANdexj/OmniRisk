#!/usr/bin/env python3

import rosbag
import numpy as np
from tqdm import tqdm
import argparse
import os
import sensor_msgs.point_cloud2 as pc2

def save_to_pcd(points, filename):
    """
    Save point cloud data directly as a PCD file
    
    Args:
        points (numpy.ndarray): point cloud data, shape (N, 3)
        filename (str): output file path
    """
    with open(filename, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def convert_bag_to_pcd(bag_file, topic_name, output_file, max_points=None, downsample=1):
    """
    Accumulate point clouds from a ROS bag and convert them to a PCD file
    
    Args:
        bag_file (str): input bag file path
        topic_name (str): point cloud topic name
        output_file (str): output PCD file path
        max_points (int, optional): max number of point clouds to process, None for all
        downsample (int, optional): downsample rate, keep one point every N
    """
    print(f"Opening bag file: {bag_file}")
    bag = rosbag.Bag(bag_file)
    
    # get the total message count for the progress bar
    msg_count = bag.get_message_count(topic_name)
    print(f"Found {msg_count} messages on topic '{topic_name}'")
    
    if max_points is not None:
        print(f"Processing at most {max_points} messages")
        msg_count = min(msg_count, max_points)
    
    # list for all point clouds
    all_points = []
    point_count = 0
    
    # iterate over point cloud messages in the bag
    print("Extracting point cloud data...")
    for i, (_, msg, _) in enumerate(tqdm(bag.read_messages(topics=[topic_name]), total=msg_count)):
        if max_points is not None and i >= max_points:
            break
            
        # convert the point cloud message to a numpy array
        try:
            # try PointCloud2 messages
            if hasattr(msg, 'data') and hasattr(msg, 'fields'):
                # PointCloud2
                points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"))))
            elif hasattr(msg, 'points'):
                # PointCloud (legacy format)
                points = np.array([[p.x, p.y, p.z] for p in msg.points])
            else:
                print(f"Warning: message {i} is not a supported point cloud format, skipping")
                continue
                
            # apply downsampling
            if downsample > 1:
                points = points[::downsample]
                
            # append this batch to the accumulated list
            all_points.append(points)
            point_count += len(points)
                
        except Exception as e:
            print(f"Error processing message {i}: {e}")
    
    bag.close()
    
    # merge all point clouds
    print(f"Merging {len(all_points)} point clouds, {point_count} points in total...")
    if not all_points:
        print("Error: no valid point cloud data found")
        return False
        
    all_points_array = np.vstack(all_points)
    print(f"Accumulated point cloud size: {all_points_array.shape}")
    
    # create and save the PCD file
    print(f"Creating PCD file: {output_file}")
    save_to_pcd(all_points_array, output_file)
    
    print(f"Saved point cloud data to {output_file}")
    print(f"PCD file contains {len(all_points_array)} points")
    
    return True 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accumulate point clouds from a ROS bag and convert them to a PCD file')
    parser.add_argument('bag_file', help='input ROS bag file path')
    parser.add_argument('topic_name', help='point cloud topic name')
    parser.add_argument('--output', '-o', help='output PCD file path (default: bag name with a .pcd suffix)')
    parser.add_argument('--max-points', '-m', type=int, help='max number of point cloud messages to process')
    parser.add_argument('--downsample', '-d', type=int, default=1, help='downsample rate (default: 1, no downsampling)')
    
    args = parser.parse_args()
    
    # if no output file is given, use the input name with a .pcd suffix
    if args.output is None:
        base_name = os.path.splitext(args.bag_file)[0]
        args.output = f"{base_name}.pcd"
    
    convert_bag_to_pcd(args.bag_file, args.topic_name, args.output, args.max_points, args.downsample)

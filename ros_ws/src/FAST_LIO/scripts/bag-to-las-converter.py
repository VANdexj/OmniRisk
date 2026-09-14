#!/usr/bin/env python3

import rosbag
import numpy as np
import laspy
from tqdm import tqdm
import argparse
import os
import sensor_msgs.point_cloud2 as pc2

def convert_bag_to_las(bag_file, topic_name, output_file, max_points=None, downsample=1):
    """
    Accumulate point clouds from a ROS bag and convert them to a LAS file
    
    Args:
        bag_file (str): input bag file path
        topic_name (str): point cloud topic name
        output_file (str): output LAS file path
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
    
    # create the LAS file
    print(f"Creating LAS file: {output_file}")
    las = laspy.create(file_version="1.4", point_format=7)
    
    # set coordinates
    las.x = all_points_array[:, 0]
    las.y = all_points_array[:, 1]
    las.z = all_points_array[:, 2]
    
    # save the LAS file
    las.write(output_file)
    
    print(f"Saved point cloud data to {output_file}")
    print(f"LAS file contains {len(las.points)} points")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accumulate point clouds from a ROS bag and convert them to a LAS file')
    parser.add_argument('bag_file', help='input ROS bag file path')
    parser.add_argument('topic_name', help='point cloud topic name')
    parser.add_argument('--output', '-o', help='output LAS file path (default: bag name with a .las suffix)')
    parser.add_argument('--max-points', '-m', type=int, help='max number of point cloud messages to process')
    parser.add_argument('--downsample', '-d', type=int, default=1, help='downsample rate (default: 1, no downsampling)')
    
    args = parser.parse_args()
    
    # if no output file is given, use the input name with a .las suffix
    if args.output is None:
        base_name = os.path.splitext(args.bag_file)[0]
        args.output = f"{base_name}.las"
    
    convert_bag_to_las(args.bag_file, args.topic_name, args.output, args.max_points, args.downsample)

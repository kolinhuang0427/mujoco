#!/usr/bin/env python3
"""
Render Expert Data on MyoLegs Model
Loads expert motion capture data and animates it on the MyoLegs skeleton in MuJoCo.
"""

import numpy as np
import pandas as pd
import pickle
import time
import mujoco
import mujoco.viewer
import os
import argparse
from typing import Optional


class ExpertDataRenderer:
    """
    Renderer for expert motion capture data on MyoLegs model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the renderer with MyoLegs model."""
        
        # Load MyoLegs model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model", "myolegs", "myolegs.xml")
            if not os.path.exists(model_path):
                model_path = "../model/myolegs/myolegs.xml"
        
        print(f"Loading MyoLegs model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Joint names that match the expert data
        self.joint_names = [
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'
        ]
        
        # Create mapping from joint names to qpos indices
        self.joint_qpos_map = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.joint_qpos_map[joint_name] = qpos_idx
                print(f"Joint '{joint_name}' -> qpos[{qpos_idx}]")
            else:
                print(f"Warning: Joint '{joint_name}' not found in model")
        
        print(f"✅ MyoLegs model loaded with {len(self.joint_qpos_map)} joints mapped")
    
    def load_expert_data(self, data_path: str) -> pd.DataFrame:
        """Load expert data from pickle file."""
        print(f"Loading expert data from: {data_path}")
        
        with open(data_path, 'rb') as f:
            expert_data = pickle.load(f)
        
        if 'qpos' not in expert_data:
            raise ValueError("Expert data must contain 'qpos' key")
        
        qpos_data = expert_data['qpos']
        print(f"✅ Loaded expert data with {len(qpos_data)} frames")
        print(f"   Available joints: {list(qpos_data.columns)}")
        
        return qpos_data
    
    def set_pose_from_expert_data(self, frame_data: pd.Series):
        """Set the model pose from a single frame of expert data."""
        
        # Reset to default pose first
        mujoco.mj_resetData(self.model, self.data)
        
        # Set root body position (keep standing height)
        self.data.qpos[0] = 0.0  # x position
        self.data.qpos[1] = 0.0  # y position  
        self.data.qpos[2] = 0.98  # z position (standing height)
        
        # Set root orientation (upright)
        self.data.qpos[3] = 1.0  # quaternion w
        self.data.qpos[4] = 0.0  # quaternion x
        self.data.qpos[5] = 0.0  # quaternion y
        self.data.qpos[6] = 0.0  # quaternion z
        
        # Set joint angles from expert data
        for joint_name in self.joint_names:
            if joint_name in frame_data and joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                joint_angle = frame_data[joint_name]
                
                # Convert to float and handle NaN values
                if pd.notna(joint_angle):
                    self.data.qpos[qpos_idx] = float(joint_angle)
        
        # Zero velocities
        self.data.qvel[:] = 0.0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
    
    def render_expert_data(self, data_path: str, fps: float = 30.0, loop: bool = True, 
                          start_frame: int = 0, end_frame: Optional[int] = None):
        """
        Render the expert data animation.
        
        Args:
            data_path: Path to expert data pickle file
            fps: Frames per second for animation
            loop: Whether to loop the animation
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
        """
        
        # Load expert data
        expert_data = self.load_expert_data(data_path)
        
        # Determine frame range
        total_frames = len(expert_data)
        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)
        
        print(f"🎬 Rendering frames {start_frame} to {end_frame-1} at {fps} FPS")
        print(f"   Total duration: {(end_frame - start_frame) / fps:.1f} seconds")
        
        # Setup viewer
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        
        frame_time = 1.0 / fps
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                # Calculate current frame
                current_frame = start_frame + (frame_count % (end_frame - start_frame))
                frame_data = expert_data.iloc[current_frame]
                
                # Set pose from expert data
                self.set_pose_from_expert_data(frame_data)
                
                # Update viewer
                viewer.sync()
                
                # Print progress every second
                if frame_count % fps == 0:
                    progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                    print(f"Frame {current_frame}/{end_frame-1} ({progress:.1f}%)")
                
                frame_count += 1
                
                # If not looping and reached the end, break
                if not loop and current_frame >= end_frame - 1:
                    print("Animation complete (no loop)")
                    break
                
                # Frame rate control
                elapsed = time.time() - start_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n🛑 Animation stopped by user")
        
        finally:
            viewer.close()
    
    def export_animation_frames(self, data_path: str, output_dir: str, 
                               start_frame: int = 0, end_frame: Optional[int] = None,
                               skip_frames: int = 1):
        """
        Export animation frames as images.
        
        Args:
            data_path: Path to expert data pickle file
            output_dir: Directory to save images
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            skip_frames: Save every N frames (1 = save all frames)
        """
        
        # Load expert data
        expert_data = self.load_expert_data(data_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine frame range
        total_frames = len(expert_data)
        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)
        
        print(f"📸 Exporting frames {start_frame} to {end_frame-1} (every {skip_frames} frames)")
        
        # Setup renderer
        renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        exported_count = 0
        for frame_idx in range(start_frame, end_frame, skip_frames):
            frame_data = expert_data.iloc[frame_idx]
            
            # Set pose from expert data
            self.set_pose_from_expert_data(frame_data)
            
            # Render frame
            renderer.update_scene(self.data)
            pixels = renderer.render()
            
            # Save image
            from PIL import Image
            image = Image.fromarray(pixels)
            image_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            image.save(image_path)
            
            exported_count += 1
            if exported_count % 10 == 0:
                print(f"Exported {exported_count} frames...")
        
        print(f"✅ Exported {exported_count} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render expert motion capture data on MyoLegs model")
    parser.add_argument("--data", type=str, default="data/expert_data.pkl",
                       help="Path to expert data pickle file")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to MyoLegs model XML file (auto-detected if not specified)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Animation frame rate")
    parser.add_argument("--loop", action="store_true", default=True,
                       help="Loop the animation")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                       help="Don't loop the animation")
    parser.add_argument("--start", type=int, default=0,
                       help="Starting frame index")
    parser.add_argument("--end", type=int, default=None,
                       help="Ending frame index")
    parser.add_argument("--export", type=str, default=None,
                       help="Export frames to directory instead of interactive viewing")
    parser.add_argument("--skip", type=int, default=1,
                       help="For export: save every N frames")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Expert data file not found: {args.data}")
        print("   Please run convert_theia_data_labels.py first to generate expert data")
        return
    
    # Create renderer
    renderer = ExpertDataRenderer(args.model)
    
    if args.export:
        # Export mode
        renderer.export_animation_frames(
            args.data, args.export, args.start, args.end, args.skip
        )
    else:
        # Interactive viewing mode
        print("\n🎬 Starting expert data animation...")
        print("   Press Ctrl+C to stop")
        renderer.render_expert_data(
            args.data, args.fps, args.loop, args.start, args.end
        )


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Render Expert Data on Torque Skeleton Model
Loads expert motion capture data and animates it on the torque skeleton in MuJoCo.
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


class TorqueSkeletonExpertDataRenderer:
    """
    Renderer for expert motion capture data on torque skeleton model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the renderer with torque skeleton model."""
        
        # Load torque skeleton model
        if model_path is None:
            # Try different possible paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "model", "torque_skeleton", "humanoid_torque.xml"),
                "model/torque_skeleton/humanoid_torque.xml",
                "../model/torque_skeleton/humanoid_torque.xml",
                os.path.join(os.getcwd(), "model", "torque_skeleton", "humanoid_torque.xml"),
            ]
            
            model_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    model_path = abs_path
                    break
            
            if model_path is None:
                raise FileNotFoundError("Could not find humanoid_torque.xml model file")
        
        print(f"Loading torque skeleton model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Joint names that match the expert data (same as in walk_env.py)
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
        
        print(f"✅ Torque skeleton model loaded with {len(self.joint_qpos_map)} joints mapped")
        
        # Target height for standing (same as in walk_env.py)
        self.target_height = 0.975
    
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
        self.data.qpos[2] = self.target_height  # z position (standing height)
        
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
        
        # Zero motor controls
        self.data.ctrl[:] = 0.0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
    
    def render_expert_data(self, data_path: str, fps: float = 30.0, loop: bool = True, 
                          start_frame: int = 0, end_frame: Optional[int] = None,
                          speed_multiplier: float = 1.0):
        """
        Render the expert data animation.
        
        Args:
            data_path: Path to expert data pickle file
            fps: Frames per second for animation
            loop: Whether to loop the animation
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            speed_multiplier: Animation speed multiplier (1.0 = normal speed)
        """
        
        # Load expert data
        expert_data = self.load_expert_data(data_path)
        
        # Determine frame range
        total_frames = len(expert_data)
        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)
        
        print(f"🎬 Rendering frames {start_frame} to {end_frame-1} at {fps} FPS")
        print(f"   Total duration: {(end_frame - start_frame) / fps / speed_multiplier:.1f} seconds")
        print(f"   Speed multiplier: {speed_multiplier}x")
        
        # Setup viewer
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        
        frame_time = 1.0 / (fps * speed_multiplier)
        frame_count = 0
        
        print("\n💡 Controls:")
        print("   - Press Ctrl+C to stop")
        print("   - Use mouse to rotate view")
        print("   - Mouse wheel to zoom")
        
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
                if frame_count % int(fps * speed_multiplier) == 0:
                    progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                    cycle_time = (current_frame - start_frame) / fps
                    print(f"Frame {current_frame}/{end_frame-1} ({progress:.1f}%) - Cycle time: {cycle_time:.1f}s")
                
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
    
    def analyze_expert_data(self, data_path: str):
        """Analyze the expert data and print statistics."""
        
        expert_data = self.load_expert_data(data_path)
        
        print(f"\n📊 Expert Data Analysis:")
        print(f"   Total frames: {len(expert_data)}")
        print(f"   Duration at 30 FPS: {len(expert_data) / 30:.1f} seconds")
        print(f"   Available joints: {len(expert_data.columns)}")
        
        print(f"\n📈 Joint Range Statistics:")
        for joint_name in self.joint_names:
            if joint_name in expert_data.columns:
                joint_data = expert_data[joint_name]
                print(f"   {joint_name:20}: "
                      f"min={joint_data.min():6.3f}, "
                      f"max={joint_data.max():6.3f}, "
                      f"mean={joint_data.mean():6.3f}, "
                      f"std={joint_data.std():6.3f}")
        
        # Check for missing joints
        missing_joints = set(self.joint_names) - set(expert_data.columns)
        if missing_joints:
            print(f"\n⚠️  Missing joints in expert data: {missing_joints}")
        
        mapped_joints = set(self.joint_names) & set(expert_data.columns)
        print(f"\n✅ Joints that will be animated: {len(mapped_joints)}/{len(self.joint_names)}")
    
    def export_animation_frames(self, data_path: str, output_dir: str, 
                               start_frame: int = 0, end_frame: Optional[int] = None,
                               skip_frames: int = 1, resolution: tuple = (1280, 720)):
        """
        Export animation frames as images.
        
        Args:
            data_path: Path to expert data pickle file
            output_dir: Directory to save images
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            skip_frames: Save every N frames (1 = save all frames)
            resolution: Image resolution (width, height)
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
        print(f"   Resolution: {resolution[0]}x{resolution[1]}")
        
        # Setup renderer
        renderer = mujoco.Renderer(self.model, height=resolution[1], width=resolution[0])
        
        exported_count = 0
        for frame_idx in range(start_frame, end_frame, skip_frames):
            frame_data = expert_data.iloc[frame_idx]
            
            # Set pose from expert data
            self.set_pose_from_expert_data(frame_data)
            
            # Render frame
            renderer.update_scene(self.data)
            pixels = renderer.render()
            
            # Save image
            try:
                from PIL import Image
                image = Image.fromarray(pixels)
                image_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                image.save(image_path)
                exported_count += 1
                
                if exported_count % 10 == 0:
                    print(f"Exported {exported_count} frames...")
                    
            except ImportError:
                print("❌ PIL not available for image export. Install with: pip install Pillow")
                break
        
        print(f"✅ Exported {exported_count} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render expert motion capture data on torque skeleton model")
    parser.add_argument("--data", type=str, default="data/expert_data.pkl",
                       help="Path to expert data pickle file")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to torque skeleton model XML file (auto-detected if not specified)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Animation frame rate")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Animation speed multiplier (1.0 = normal speed)")
    parser.add_argument("--loop", action="store_true", default=True,
                       help="Loop the animation")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                       help="Don't loop the animation")
    parser.add_argument("--start", type=int, default=0,
                       help="Starting frame index")
    parser.add_argument("--end", type=int, default=None,
                       help="Ending frame index")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze expert data and exit (no animation)")
    parser.add_argument("--export", type=str, default=None,
                       help="Export frames to directory instead of interactive viewing")
    parser.add_argument("--skip", type=int, default=1,
                       help="For export: save every N frames")
    parser.add_argument("--resolution", type=str, default="1280x720",
                       help="For export: image resolution (WIDTHxHEIGHT)")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Expert data file not found: {args.data}")
        print("   Please ensure expert_data.pkl exists in the data/ directory")
        return
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"❌ Invalid resolution format: {args.resolution}. Use WIDTHxHEIGHT (e.g., 1280x720)")
        return
    
    # Create renderer
    try:
        renderer = TorqueSkeletonExpertDataRenderer(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    if args.analyze:
        # Analysis mode
        renderer.analyze_expert_data(args.data)
    elif args.export:
        # Export mode
        renderer.export_animation_frames(
            args.data, args.export, args.start, args.end, args.skip, resolution
        )
    else:
        # Interactive viewing mode
        print("\n🎬 Starting expert data animation...")
        print("   This shows the raw motion capture data on the torque skeleton")
        renderer.render_expert_data(
            args.data, args.fps, args.loop, args.start, args.end, args.speed
        )


if __name__ == "__main__":
    main() 
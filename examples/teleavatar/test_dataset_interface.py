#!/usr/bin/env python3
"""
Simple test script for dataset_interface.py
Tests basic functionality without requiring a policy server.
"""

import logging
import sys
sys.path.append('/home/caslx/Robotics/openpi')

from examples.teleavatar import dataset_interface


def main():
    """Test the dataset interface."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    
    print("=" * 60)
    print("Testing Teleavatar Dataset Interface")
    print("=" * 60)
    
    # Create interface
    dataset_path = "/home/caslx/Robotics/openpi/datasets"
    episode_index = 0
    
    print(f"\n1. Initializing dataset interface...")
    print(f"   Dataset path: {dataset_path}")
    print(f"   Episode: {episode_index}")
    
    try:
        interface = dataset_interface.TeleavatarDatasetInterface(
            dataset_path=dataset_path,
            episode_index=episode_index,
            start_frame=0,
        )
        print("   ✓ Interface initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize interface: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Wait for data (always succeeds for dataset)
    print("\n2. Waiting for initial data...")
    if not interface.wait_for_initial_data():
        print("   ✗ Failed to load initial data")
        return 1
    print("   ✓ Initial data ready")
    
    # Get episode info
    print("\n3. Episode information:")
    info = interface.get_episode_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Read first few frames
    print("\n4. Reading first 5 frames:")
    for i in range(5):
        obs = interface.get_observation()
        if obs is None:
            print(f"   Frame {i}: No more data (reached end of episode)")
            break
        
        print(f"\n   Frame {i}:")
        print(f"     State shape: {obs['state'].shape}")
        print(f"     State (positions, first 8): {obs['state'][:8].tolist()}")
        print(f"     State (velocities, first 8): {obs['state'][16:24].tolist()}")
        print(f"     State (efforts, first 8): {obs['state'][32:40].tolist()}")
        print(f"     Images:")
        for cam_name, img in obs['images'].items():
            print(f"       {cam_name}: shape={img.shape}, dtype={img.dtype}, "
                  f"range=[{img.min()}, {img.max()}]")
    
    # Test reset functionality
    print("\n5. Testing reset to frame 0...")
    interface.reset()
    obs = interface.get_observation()
    if obs is not None:
        print("   ✓ Reset successful, read first frame again")
        print(f"     State (first 8 values): {obs['state'][:8].tolist()}")
    else:
        print("   ✗ Reset failed")
        return 1
    
    # Test reset to different episode (if available)
    print("\n6. Testing reset to episode 1 (if available)...")
    try:
        interface.reset(episode_index=1, start_frame=0)
        info = interface.get_episode_info()
        print(f"   ✓ Reset to episode {info['episode_index']}")
        print(f"     Task: {info['task']}")
        print(f"     Length: {info['episode_length']} frames")
        
        obs = interface.get_observation()
        if obs is not None:
            print(f"     State (first 8 values): {obs['state'][:8].tolist()}")
    except Exception as e:
        print(f"   ⚠ Could not reset to episode 1: {e}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


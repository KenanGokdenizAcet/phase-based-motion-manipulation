# Phase-Based Motion Manipulation

This project implements phase-based motion manipulation techniques in Unity using the Built-in Render Pipeline.

## Setup Instructions

1. Ensure you're using the Built-in Render Pipeline (not URP)
2. Add the `MotionMagnificationProcess` component to your camera
3. In the inspector, configure the following:
   - Assign the required shader materials
   - Set the `Phase Scale` parameter (this controls the motion magnification factor)
   - Adjust other parameters as needed

## Requirements

- Unity 2022.3 LTS or newer
- Built-in Render Pipeline
- Graphics card that supports compute shaders

## Usage

The `Phase Scale` parameter in the `MotionMagnificationProcess` component controls the strength of the motion magnification effect. Higher values will result in more pronounced motion amplification.

## Troubleshooting

If you experience issues:
1. Verify you're using the Built-in Render Pipeline
2. Check that all shader materials are properly assigned in the inspector
3. Ensure your graphics card supports compute shaders

import mediapipe as mp

print(f"MediaPipe version: {mp.__version__}")
print(f"Has solutions: {hasattr(mp, 'solutions')}")
if hasattr(mp, 'solutions'):
    print(f"Pose in solutions: {hasattr(mp.solutions, 'pose')}")

# Test tạo pose object
try:
    pose = mp.solutions.pose.Pose()
    print("✅ Successfully created Pose object")
    pose.close()
except Exception as e:
    print(f"❌ Error creating Pose: {e}")
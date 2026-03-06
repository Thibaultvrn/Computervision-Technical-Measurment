"""Script to detect available cameras and find Camo virtual webcam."""

import cv2


def list_available_cameras(max_index: int = 10) -> list:
    """
    Detect all available camera indices.
    
    Args:
        max_index: Maximum index to check
        
    Returns:
        List of available camera indices with their properties
    """
    available = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow for Windows
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to read a frame to confirm it works
            ret, frame = cap.read()
            
            available.append({
                "index": i,
                "width": width,
                "height": height,
                "fps": fps,
                "working": ret
            })
            cap.release()
    
    return available


def test_camera(index: int) -> None:
    """
    Open a preview window to test a specific camera.
    
    Args:
        index: Camera index to test
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        return
    
    print(f"Testing camera {index} - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        cv2.imshow(f"Camera {index} Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Detecting available cameras...")
    print("-" * 40)
    
    cameras = list_available_cameras()
    
    if not cameras:
        print("No cameras found!")
    else:
        for cam in cameras:
            status = "OK" if cam["working"] else "NO SIGNAL"
            print(f"Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps [{status}]")
        
        print("-" * 40)
        print("\nTo test a camera, enter its index (or 'q' to quit):")
        
        while True:
            choice = input("> ").strip()
            if choice.lower() == 'q':
                break
            try:
                idx = int(choice)
                test_camera(idx)
            except ValueError:
                print("Enter a valid number or 'q'")

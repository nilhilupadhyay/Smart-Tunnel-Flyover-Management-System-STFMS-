import easyocr
import re
import numpy as np

# Global variable to hold the initialized model
ocr_reader = None

def initialize_anpr():
    """
    Loads the EasyOCR model into memory.
    """
    global ocr_reader
    if ocr_reader is not None:
        return # Already loaded

    print("[INFO] Loading EasyOCR model (this may take a moment)...")
    
    # (FIXED) Removed all allowlist arguments to prevent version crashes.
    # We will accept the [WARN] and continue.
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        print(f"[ERROR] Failed to load EasyOCR: {e}")
        return
            
    print("[INFO] EasyOCR model ready.")

def get_plate_from_frame(frame: np.ndarray, box: np.ndarray) -> str:
    """
    Detects and returns a license plate string from a cropped image.
    Uses "smart cropping" to isolate the plate area.
    """
    global ocr_reader
    if ocr_reader is None:
        print("[ERROR] ANPR model not initialized. Call initialize_anpr() first.")
        return "N/A"

    try:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # --- (NEW) "Smart Crop" Logic ---
        # Instead of the whole box, let's crop the bottom 40%
        # and the middle 80% (width-wise).
        
        box_height = y2 - y1
        box_width = x2 - x1
        
        # Crop Y (Vertical): Bottom 40% of the car
        crop_y1 = y2 - int(box_height * 0.4) 
        crop_y2 = y2
        
        # Crop X (Horizontal): Middle 80% of the car
        crop_x1 = x1 + int(box_width * 0.1) 
        crop_x2 = x2 - int(box_width * 0.1)
        
        # Ensure coordinates are valid
        if crop_y1 < 0: crop_y1 = 0
        if crop_x1 < 0: crop_x1 = 0
        
        # Perform the smart crop
        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        # --- End of Smart Crop ---

        # Check if crop is too small (avoids errors)
        if plate_crop.shape[0] < 20 or plate_crop.shape[1] < 40:
            return "N/A (Too Small)"
        
        # Run OCR on the *small, focused* crop
        ocr_results = ocr_reader.readtext(plate_crop, detail=0)
        
        for text in ocr_results:
            # Clean the text: remove spaces, symbols, etc.
            text = text.upper()
            text = re.sub(r'[^A-Z0-9]', '', text) 
            
            # (FIXED) Relaxed filter for plates
            if 4 <= len(text) <= 9:
                print(f"[ANPR] Found plate: {text}")
                return text
                
    except Exception as e:
        print(f"[ANPR] Error: {e}")
    
    return "N/A"
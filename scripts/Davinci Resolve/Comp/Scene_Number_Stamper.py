import sys
import pprint 

# --- [USER CONFIGURATION] ---
# The control ID where the marker name will be written
TEXT_CONTROL_ID = "SceneNumber" 
# Use 0 for exact alignment with the marker frame (as you confirmed)
FRAME_OFFSET = 0 
# --- [END USER CONFIGURATION] ---

# --- INTERNAL CONSTANTS (Do Not Modify) ---
# Internal constant for Fusion's Text data type (FU_DataType_Text = 0x00000003)
TEXT_DATATYPE = 0x00000003 
# --- END INTERNAL CONSTANTS ---


# --- HELPER FUNCTIONS (From previous script, adapted for text) ---

def get_target_tool(comp):
    node_name = "Control_Panel"
    target_tool = comp.FindTool(node_name)
    if target_tool is None:
        print(f"\n--- ERROR: Required node '{node_name}' not found. ---", file=sys.stderr)
        return None
    return target_tool

def get_spline_tool(input_obj):
    """Retrieves the actual BezierSpline tool object connected to an input."""
    output = input_obj.GetConnectedOutput()
    if output and (output.GetTool().ID == "BezierSpline" or output.GetTool().ID == "TextSpline"):
        return output.GetTool()
    return None

def process_scene_number_text(target_tool, comp, markers_dict, marker_frame_ids):
    """
    Attempts to set keyframes by simply assigning values at specific times.
    This relies on Fusion's auto-keying behavior.
    """
    
    print(f"\nProcessing Text Control: {TEXT_CONTROL_ID} (Brute Force Mode)")
    
    # 1. Retrieve the Input
    input_obj = getattr(target_tool, TEXT_CONTROL_ID, None)
    if input_obj is None:
        try: input_obj = target_tool[TEXT_CONTROL_ID]
        except: pass
    
    if input_obj is None:
        print(f"  - ERROR: Text Control '{TEXT_CONTROL_ID}' not found. Skipping.", file=sys.stderr)
        return

    # 2. Enable Keyframing (The Critical Step?)
    # Sometimes simply connecting it to a BezierSpline manually works, 
    # but via script we try to 'touch' it at different times.
    
    # NOTE: We are NOT clearing the modifier. If one exists (from your manual test), we want to use it.
    # If one doesn't exist, this loop attempts to force one.

    count = 0
    
    # 3. Loop and Assign
    for frame_id in marker_frame_ids:
        
        marker_data = markers_dict.get(float(frame_id)) 
        if marker_data is None: continue 

        marker_name = marker_data.get("name", f"Marker_{frame_id}")
        corrected_id = float(frame_id) - FRAME_OFFSET
        
        # --- THE BRUTE FORCE ASSIGNMENT ---
        # Syntax: Input[Time] = Value
        try:
            input_obj[corrected_id] = marker_name
            count += 1
        except Exception as e:
            print(f"  - Failed to set key at frame {corrected_id}: {e}")

    print(f"  - Attempted to direct-assign {count} keys.")
    print("  - Check if the 'Key' icon is now active on the SceneNumber control.")

def main():
    try:
        global resolve, fusion
        resolve = resolve
        fusion = fusion
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        comp = fusion.GetCurrentComp()
        
        if not (project and timeline and comp): raise Exception("API not loaded.")
    except Exception as e:
        print(f"Error setup: {e}")
        sys.exit()
    
    print("--- Scene Number Stamper (V6 - Brute Force) ---")

    target_tool = get_target_tool(comp)
    if target_tool:
        markers_dict = timeline.GetMarkers()
        if markers_dict:
            marker_frame_ids = sorted([int(frame_id) for frame_id in markers_dict.keys()])
            
            comp.Lock()
            try:
                process_scene_number_text(target_tool, comp, markers_dict, marker_frame_ids)
            finally:
                comp.Unlock()
    
    print("\n--- Script Complete ---")

if __name__ == "__main__":
    main()
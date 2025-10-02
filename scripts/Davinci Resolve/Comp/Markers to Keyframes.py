import sys
# Used for safe string representation of values in diagnostics
from pprint import pprint 
import os

# --- [USER CONFIGURATION] ---
FRAME_OFFSET = 1 
KF_STEP = 2 # The flag value for 'Step' interpolation.
# Add or remove control IDs here, using the desired mode/values.
controls_to_keyframe = {
    "MaxDisparity": {"mode": "STATIC_VALUE",},    
    "Convergence": {"mode": "STATIC_VALUE",},
    "FrontGamma": {"mode": "STATIC_VALUE",},
    "Bias": {"mode": "STATIC_VALUE",},
    "Overlap": {"mode": "STATIC_VALUE",},
}
# --- [END USER CONFIGURATION] ---


# --- HELPER FUNCTIONS ---
def clean_and_add_spline(tool, input_name, comp_start_time):
    """
    Clears old modifiers, gets static value, and adds a fresh spline at start time.
    (Reusable function from the Exporter script)
    """
    input_obj = None
    
    # Use the robust look-up sequence from the Exporter
    try:
        input_obj = tool[input_name]
    except (AttributeError, KeyError):
        input_obj = getattr(tool, input_name, None)

    if input_obj is None:
        print(f"  - ERROR: Control '{input_name}' not found on node. Skipping.", file=sys.stderr)
        return None, None
    
    # 1. Disconnect any existing modifier to get the underlying static value
    initial_value = input_obj[comp_start_time]
    
    if input_obj.GetConnectedOutput() is not None:
        input_obj.ConnectTo(None) 
        initial_value = input_obj[comp_start_time] 
        print(f"  - Cleared existing modifier. Static Value: {initial_value:.4f}")
    
    # 2. Add a new BezierSpline
    success = tool.AddModifier(input_name, "BezierSpline")
    if not success:
         print(f"  - WARNING: Failed to add BezierSpline to '{input_name}'. Skipping.", file=sys.stderr)
         return None, None
         
    # 3. Clean up the artifact keyframe
    input_obj[comp_start_time] = None 
    
    return initial_value, input_obj

def confirm_keyframe_overwrite(comp, controls_to_keyframe, marker_frame_ids):
    """Displays the overwrite warning dialog and exits if the user cancels."""
    controls_list = "\n - ".join(controls_to_keyframe.keys())
    marker_frames_str = ", ".join(map(str, marker_frame_ids))

    dialog_message = ("Are you sure you want to **OVERWRITE/APPEND EXISTING KEYFRAMES** "
                      "for the following controls with Step-In interpolation at marker positions?")

    dialog_controls = {
        1: {1: "WarningLabel", "Name": dialog_message, 2: "Text", "ReadOnly": True, "Lines": 5, "Wrap": True, "LINKID_DataType": "Number"},
        2: {1: "ControlsAffected", "Name": "Controls to be Modified:", 2: "Text", "ReadOnly": True, "Lines": len(controls_to_keyframe) + 1, "Default": "\n - " + controls_list, "LINKID_DataType": "Number"},
        3: {1: "FramesInserted", "Name": f"Frames to be Modified ({len(marker_frame_ids)} total):", 2: "Text", "ReadOnly": True, "Lines": 3, "Default": marker_frames_str, "LINKID_DataType": "Number"},
        4: {1: { "MBTNC_AddButton": "RUN SCRIPT" }, 2: { "MBTNC_AddButton": "CANCEL" }, "LINKID_DataType": "Number", "INPID_InputControl": "MultiButtonControl", "MBTNC_ShowBasicButton": False, "MBTNC_ShowName": False, "MBTNC_StretchToFit": True, "LINKS_Name": "Select Action", "INP_Default": 0, "INP_Passive": False, "INP_External": False},
    }

    dialog_result = comp.AskUser("!! WARNING: Keyframe Data Will Be Modified !!", dialog_controls)
    
    if dialog_result is None or dialog_result.get(4) == 1.0: 
        print("\n--- OPERATION CANCELED BY USER. NO KEYFRAMES WERE MODIFIED. ---")
        sys.exit()
    
    print("\n--- Confirmation received. Proceeding with keyframe modification. ---")

def get_spline_tool(input_obj):
    """Retrieves the actual BezierSpline tool object connected to an input."""
    output = input_obj.GetConnectedOutput()
    if output and output.GetTool().ID == "BezierSpline":
        return output.GetTool()
    return None

def get_target_tool(comp):
    """Tries to get the active tool. If not valid, asks the user."""
    active_tool = comp.ActiveTool
    if active_tool is not None:
        print(f"INFO: Using currently selected node: '{active_tool.Name}'")
        return active_tool
    
    dialog_controls = {
        1: { 1: "NodeName", "Name": "Enter Control Node Name", 2: "Text", "Default": "Control_Panel", "LINKID_DataType": "Text"},
    }
    dialog_result = comp.AskUser("Select Node for Keyframing", dialog_controls)
    if dialog_result is None:
        print("\n--- NODE SELECTION CANCELED BY USER. ---")
        return None
    
    node_name = dialog_result.get(1)
    target_tool = comp.FindTool(node_name)
    if target_tool is None:
        print(f"\n--- ERROR: Node named '{node_name}' not found. ---")
        return None
        
    print(f"INFO: Found node: '{target_tool.Name}'")
    return target_tool

def process_keyframe_logic(target_tool, comp, marker_frame_ids, controls_to_keyframe):
    """Handles the reading of splines, setting of step-in keyframes, and writing back."""
    
    # ... (Setup comp_start_time, comp.CurrentTime = comp_start_time) ...
    
    comp.Lock()
    try:
        for control_id, config in controls_to_keyframe.items():
            print(f"\nProcessing control: {control_id}")

            input_obj = getattr(target_tool, control_id, None) 
            if input_obj is None:
                print(f"  - ERROR: Control '{control_id}' not found on node. Skipping.", file=sys.stderr)
                continue
            
            # Get the existing spline tool, or create a new one
            spline_tool = get_spline_tool(input_obj)
            if spline_tool is None:
                # Force read static value before adding modifier
                initial_value = input_obj[comp.CurrentTime]
                target_tool.AddModifier(control_id, "BezierSpline")
                spline_tool = get_spline_tool(input_obj)
                existing_keyframes = {} # New spline has no keys yet
                # Clean up the initial keyframe created by AddModifier
                input_obj[comp.CurrentTime] = None 
                print("  - Created new BezierSpline.")
            else:
                existing_keyframes = spline_tool.GetKeyFrames()
                print(f"  - Read {len(existing_keyframes)} existing keyframes.")
            
            # Now we have the spline_tool and a dictionary of existing_keyframes
            final_keyframes = existing_keyframes.copy()
            
            # Iterate through markers to ADD/OVERWRITE new keyframes
            value_toggle = True
            initial_value = input_obj[comp.CurrentTime] # Read current value for STATIC mode
            
            for frame_id in marker_frame_ids:
                corrected_id = frame_id
                if frame_id > 0: corrected_id = frame_id - FRAME_OFFSET

                # CALCULATE VALUE
                current_value = initial_value 
                if config["mode"] == "ALTERNATE_VALUES":
                    val_low = config.get("value_low", 0.0); val_high = config.get("value_high", 1.0)
                    current_value = val_high if value_toggle else val_low
                    value_toggle = not value_toggle
                
                # --- CREATE THE NEW KEYFRAME DATA STRUCTURE ---
                # Key: Frame ID (corrected_id)
                # Value: List containing Value and Flags Dictionary
                final_keyframes[float(corrected_id)] = {
                    1: current_value,
                    "Flags": {
                        "Step": True # Sets the Step-In interpolation
                    }
                }
                
            # --- WRITE THE FINAL SPLINE DATA ---
            # SetKeyFrames() expects a dictionary structured as {frame_time: {1: value, "Flags":{...}}}
            spline_tool.SetKeyFrames(final_keyframes)
            
            print(f"  - Appended {len(marker_frame_ids)} Step-In keyframes.")
            
    finally:
        comp.Unlock()

def main():
    
    # --- API Setup ---
    try:
        global resolve, fusion
        resolve = resolve
        fusion = fusion
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        comp = fusion.GetCurrentComp()
        
        if not (project and timeline and comp):
            raise Exception("Resolve Project/Timeline/Fusion Comp not loaded/found.")
            
    except Exception as e:
        print(f"Error during API setup: {e}", file=sys.stderr)
        sys.exit()
    
    print("--- Markers To Keyframes (Selected Node) ---")

    # 1. GET THE TARGET NODE
    target_tool = get_target_tool(comp)
    if target_tool is None:
        sys.exit()
    
    # Check if any controls are configured before proceeding
    if not controls_to_keyframe:
        print("WARNING: No controls configured to keyframe. Exiting.")
        sys.exit()
    
    # 2. GET MARKER DATA
    markers_dict = timeline.GetMarkers()
    if not markers_dict:
        print("No Timeline Markers found. Aborting.")
        sys.exit()

    marker_frame_ids = sorted([int(frame_id) for frame_id in markers_dict.keys()])
    
    # 3. SETUP COMP TIME
    comp_start_time = comp.GetAttrs()["COMPN_GlobalStart"]
    comp.CurrentTime = comp_start_time
    
    # 4. OVERWRITE WARNING DIALOG
    confirm_keyframe_overwrite(comp, controls_to_keyframe, marker_frame_ids)
    
    # 5. EXECUTE KEYFRAME LOGIC (Destructive)
    
    # CRITICAL: Unlock the tool before modifying attributes
    try:
        if target_tool.GetAttrs().get("TOOLB_Locked") == True:
            print(f"INFO: Unlocking tool '{target_tool.Name}' before modification.")
            target_tool.SetAttrs( { "TOOLB_Locked": False } )
    except Exception as e:
        print(f"WARNING: Could not unlock tool. Error: {e}", file=sys.stderr)

    comp.Lock()
    try:
        for control_id, config in controls_to_keyframe.items():
            print(f"\nProcessing control: {control_id}")

            # --- INPUT OBJECT RETRIEVAL ---
            # Robustly find the input object using the exposed tool index
            input_obj = getattr(target_tool, control_id, None)
            if input_obj is None:
                try: input_obj = target_tool[control_id]
                except: pass
            
            if input_obj is None:
                print(f"  - ERROR: Control '{control_id}' not found on node. Skipping.", file=sys.stderr)
                continue
                
            # --- SPLINE SETUP/READ ---
            spline_tool = get_spline_tool(input_obj)
            if spline_tool is None:
                # 1. New Spline: Force read static value, add spline tool
                initial_value = input_obj[comp_start_time] # Read static value
                target_tool.AddModifier(control_id, "BezierSpline")
                spline_tool = get_spline_tool(input_obj)
                existing_keyframes = {}
                input_obj[comp_start_time] = None # Clean up initial keyframe
                print("  - Created new BezierSpline.")
            else:
                # 2. Existing Spline: Read existing keyframes
                existing_keyframes = spline_tool.GetKeyFrames()
                initial_value = input_obj[comp_start_time] # Read value at start time
                print(f"  - Read {len(existing_keyframes)} existing keyframes. Reading start value {initial_value:.4f}")

            final_keyframes = existing_keyframes.copy()

            # Create a combined set of all frame IDs (existing keys + new markers)
            all_frame_keys = set(final_keyframes.keys())
            
            # Ensure all keys in the final set get the StepIn flag
            for frame_time in all_frame_keys:
                key_data = final_keyframes.get(frame_time)
                # Ensure the key is a dictionary (it should be)
                if isinstance(key_data, dict):
                    # Set the StepIn Flag on the existing dictionary structure
                    key_data["Flags"] = { "StepIn": True }

            value_toggle = True
            
            # --- KEYFRAME GENERATION LOOP ---
            for frame_id in marker_frame_ids:
                corrected_id = float(frame_id)
                if frame_id > 0: corrected_id = float(frame_id) - FRAME_OFFSET

                # Retrieve the full keyframe dictionary at this marker time
                key_data = final_keyframes.get(corrected_id, {})
                
                # If the keyframe didn't exist at this frame, create a basic one
                if not key_data:
                    # Read interpolated value (as before) to use as the base
                    base_value = float(input_obj[corrected_id])
                    key_data[1] = base_value 
                    # Set default flags (optional, but safe)
                    key_data["Flags"] = 0
                else:
                    # Keyframe exists: Read its current value from the structure
                    base_value = float(key_data.get(1)) 

                # --- CALCULATE VALUE (Overrides Base Value if necessary) ---
                current_value = base_value
                if config["mode"] == "ALTERNATE_VALUES":
                    val_low = config.get("value_low", 0.0); val_high = config.get("value_high", 1.0)
                    current_value = val_high if value_toggle else val_low
                    value_toggle = not value_toggle
                
                # --- APPLY STEP-IN FLAG ---
                
                # The Flags attribute is being overwritten to force StepIn
                # All other attributes (like 'RH' and 'LH') are preserved because we used key_data as the template.
                key_data[1] = current_value # Update the value
                
                # If the key didn't exist (i.e., new marker on a blank frame), set its full structure
                if corrected_id not in final_keyframes:
                    key_data["Flags"] = { "StepIn": True } # Ensure flag is set for newly created key
                    final_keyframes[corrected_id] = key_data
                
            # --- WRITE THE FINAL SPLINE DATA ---
            spline_tool.SetKeyFrames(final_keyframes)
            
            print(f"  - Appended/Overwrote {len(marker_frame_ids)} Step-In keyframes.")
            
    finally:
        comp.Unlock()

    print("\n--- Script Complete: Keyframes set on selected node. ---")

if __name__ == "__main__":
    main()
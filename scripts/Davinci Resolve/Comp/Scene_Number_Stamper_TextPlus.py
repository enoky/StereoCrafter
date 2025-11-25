#!/usr/bin/env python

# --- DaVinci Resolve Script: Scene Number Stamper (Text+ Version) ---
# Target: The currently selected 'Text+' node in Fusion.
# Action: Animates the 'StyledText' input to match Timeline Markers.

import sys

# --- [USER CONFIGURATION] ---
FRAME_OFFSET = 0 
# --- [END USER CONFIGURATION] ---

def get_active_textplus_tool(comp):
    """Gets the currently selected tool and verifies it is a Text+ node."""
    target_tool = comp.ActiveTool
    
    if target_tool is None:
        print(f"\n--- ERROR: No node selected. Please select a Text+ node. ---", file=sys.stderr)
        return None
    
    # Check internal ID to ensure it is a Text+ node
    # The internal ID for Text+ is usually "TextPlus"
    if target_tool.ID != "TextPlus":
        print(f"\n--- ERROR: Selected node '{target_tool.Name}' is not a Text+ node (ID: {target_tool.ID}). ---", file=sys.stderr)
        print("Please select a standard 'Text+' node.", file=sys.stderr)
        return None
        
    print(f"INFO: Found Text+ node: '{target_tool.Name}'")
    return target_tool

def process_styled_text(target_tool, comp, markers_dict, marker_frame_ids):
    """
    Sets the 'StyledText' input at specific frames to match marker names.
    """
    
    print(f"\nProcessing Input: StyledText")
    
    # For Text+ nodes, the main text input is "StyledText"
    input_obj = target_tool.StyledText
    
    if input_obj is None:
        print(f"  - ERROR: 'StyledText' input not found. Skipping.", file=sys.stderr)
        return    
    
    # Force enable animation on the input (only needed for Text+ script)
    # This is the code equivalent of clicking the diamond icon
    target_tool.SetAttrs({"TOOLB_Locked": False}) # Unlock tool
    input_obj.SetExpression("time") # Temporarily add expression to wake up animation
    input_obj.ConnectTo(None)       # Remove expression, leaving an empty spline path

    count = 0
    
    # Loop and Assign
    for frame_id in marker_frame_ids:
        
        marker_data = markers_dict.get(float(frame_id)) 
        if marker_data is None: continue 

        # Get the name
        marker_name = marker_data.get("name", f"Marker_{frame_id}")
        
        # Apply Offset
        corrected_id = float(frame_id) - FRAME_OFFSET
        
        # --- THE ASSIGNMENT ---
        # Syntax: TextPlus.StyledText[Time] = "String"
        try:
            input_obj[corrected_id] = marker_name
            count += 1
        except Exception as e:
            print(f"  - Failed to set key at frame {corrected_id}: {e}")

    print(f"  - Successfully stamped {count} keys into StyledText.")

def main():
    # --- API Setup ---
    try:
        global resolve, fusion
        resolve = resolve
        fusion = fusion
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        comp = fusion.GetCurrentComp()
        
        if not (project and timeline and comp): raise Exception("API not loaded.")
    except Exception as e:
        # Fallback for external execution
        try:
            import DaVinciResolveScript as dvr_script
            resolve = dvr_script.scriptapp("Resolve")
            # Re-run setup if fallback works
            project = resolve.GetProjectManager().GetCurrentProject()
            timeline = project.GetCurrentTimeline()
            comp = resolve.Fusion().GetCurrentComp()
        except:
            print(f"Error setup: {e}")
            sys.exit()
    
    print("--- Scene Number Stamper (Text+ Version) ---")

    # 1. Get Selected Text+ Node
    target_tool = get_active_textplus_tool(comp)
    
    if target_tool:
        # 2. Get Marker Data
        markers_dict = timeline.GetMarkers()
        if markers_dict:
            marker_frame_ids = sorted([int(frame_id) for frame_id in markers_dict.keys()])
            
            # 3. Run Logic
            comp.Lock()
            try:
                process_styled_text(target_tool, comp, markers_dict, marker_frame_ids)
            finally:
                comp.Unlock()
        else:
            print("No markers found on the Timeline Ruler.")
    
    print("\n--- Script Complete ---")

if __name__ == "__main__":
    main()
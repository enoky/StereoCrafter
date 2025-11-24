
# DaVinci Resolve Script: Scene Number Stamper (Text+ Version)

This script synchronizes the text content of a standard Fusion **Text+** node with the names of markers on the Resolve Timeline. It creates keyframes on the `StyledText` input, causing the text to update instantly at every marker position.

## 🚀 Overview

Ideal for creating dynamic "burn-in" overlays or scene labels that update automatically as the video plays. Instead of manually keyframing text changes for every scene, this script pulls the data directly from your Edit Page timeline markers.

**Functionality:**
*   Target: The **currently selected** `Text+` node in the Fusion composition.
*   Input: Reads all markers from the active Resolve Timeline.
*   Action: Keyframes the text content to match the Marker Name at the specific frame.

## 📋 Requirements

*   **Target Node:** You must have a **`Text+`** node selected in the Fusion Flow.
*   **Markers:** Requires markers to be placed on the **Resolve Timeline Ruler**.
*   **Execution:** Run from the **Fusion Page** scripts menu.

## 💻 Installation

1.  **Save the Script:** Save the Python code as `Scene_Number_Stamper_TextPlus.py`.
2.  **Location:** Place the file in the Fusion Comp scripts folder:
    *   **Windows:** `%AppData%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp`
    *   **Mac:** `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Comp`

## 🕹️ Usage

1.  **Edit Page:** Add markers to your timeline ruler where you want the text to change. Name the markers (e.g., "Scene 1", "Shot 4", "Review Note").
2.  **Fusion Page:** Add a `Text+` node.
3.  **Select:** Click the `Text+` node to ensure it is active/selected.
4.  **Run:** Go to **Workspace > Scripts > Comp > Scene_Number_Stamper_TextPlus**.
5.  **Result:** The text input will turn green (animated) and display the marker names in sync with playback.
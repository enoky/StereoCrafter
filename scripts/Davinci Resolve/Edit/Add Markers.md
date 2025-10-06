
# DaVinci Resolve Script: Add Timeline Cuts (`Add Markers.py`)

This script automatically analyzes the currently open timeline and places three types of markers directly onto the timeline ruler: one at the start, one at the end, and one at every video clip cut point on Video Track 1.

## 🚀 Overview

This script will save you manually adding markers at cut points.

**Functionality:**

*   **Start Marker:** Places an **Red** marker at Frame 0.
*   **Cut Markers:** Places a **Cyan** marker at the start frame of every clip on Video Track 1 (the cut point).
*   **End Marker:** Places an **Red** marker at the timeline's final boundary.
*   All markers are uniquely named (e.g., "Marker 1", "Marker 2", etc.).

## 📋 Requirements

*   **DaVinci Resolve Studio:** Must be installed and running.
*   **Python:** Python 3.6+ (64-bit) installed (Required by the Resolve API).
*   **Content:** A project must be open, and a timeline must be loaded with clips on **Video Track 1**.

## 💻 Installation

The simplest method for running this script is via the Resolve Scripts Menu.

1.  **Save the Script:** Save the final Python code (V16) as a file named `Add Markers.py`.
2.  **Place the File:** Place the script `Add Markers.py`, in one of the following location-specific folders (creating the folders if necessary).
    *   **Windows:** `%APPDATA%\Roaming\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Edit`
    *   **Mac OS X:** `/Users/<UserName>/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Edit`
    *   **Linux:** `$HOME/.local/share/DaVinciResolve/Fusion/Scripts/**Edit**`

## 🕹️ Usage

1.  **Open DaVinci Resolve Studio** and load the project/timeline you wish to mark.
2.  Ensure the **Edit Page** is active.
3.  Navigate to the **Workspace** menu.
4.  Select **Scripts** -> **Edit** (or your chosen subfolder).
5.  Click on **`Add Markers`**.

Markers will be placed instantly on the timeline ruler (above the video tracks).

---
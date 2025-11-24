# DaVinci Resolve Script: Export Timeline Markers to SRT (`Export_Markers_To_SRT.py`)

This script reads all markers from the Resolve Timeline Ruler and generates a standard `.srt` subtitle file. Each subtitle block is defined by the marker's name and its duration is set by the distance to the next marker.

## 🚀 Overview

This is an essential tool for creating review video burn-ins or transcribing scene information. It is mainly used during previewing a video to get an instant feedback on which scene number(marker) you are on.

**Functionality:**
*   **Timecode Accurate:** Uses the project's actual FPS to convert frame numbers to precise SRT timecodes (HH:MM:SS,ms).
*   **Segmented Duration:** Each subtitle (marker) begins at its frame position and ends at the frame position of the very next marker. The final marker lasts for a default of 2 seconds.
*   **Output:** Prompts the user to select a folder and saves the file automatically named `[TimelineName]_Markers.srt`.

## 📋 Requirements

*   **DaVinci Resolve Studio:** Must be installed and running.
*   **Python:** Python 3.6+ (64-bit).
*   **Markers:** Requires markers to be placed on the **Resolve Timeline Ruler**.
*   **Libraries:** Requires the standard Python `tkinter` library for the folder selection dialog (included with most Python installations).

## 💻 Installation

1.  **Save the Script:** Save the Python code as `Export_Markers_To_SRT.py`.
2.  **Place the File:** Place the script in your preferred script location (e.g., the Edit scripts folder):

    *   **Windows:** `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\**Edit**`
    *   **Mac OS X:** `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/**Edit**`

## 🕹️ Usage

1.  **Open DaVinci Resolve Studio** and load the project/timeline containing your markers.
2.  Navigate to the **Workspace** menu.
3.  Select **Scripts** -> **Edit** (or your chosen subfolder).
4.  Click on **`Export_Markers_To_SRT`**.
5.  A **"Select Folder"** dialog will open. Choose the directory where you want to save the `.srt` file.
6.  The script will save the file with the name **`[TimelineName]_Markers.srt`** to the chosen folder.

# DaVinci Resolve Script: Scene Number Stamper (`Scene_Number_Stamper.py`)

This script synchronizes a text control in a Fusion node with the names of all markers placed on the Resolve Timeline Ruler. It sets stepped keyframes to instantly update the text on-screen at every marker position.

## 🚀 Overview

The script is used to create a visual "stamp" on the video that reflects the current timeline marker's name. This is extremely useful for generating burn-in video for review, allowing easy identification of specific points during playback analysis.

**Functionality:**
*   Reads the name of every marker from the Resolve Timeline.
*   Creates stepped keyframes on the target Fusion text control.
*   The text instantly updates at each marker frame to show the marker's name.

## 📋 Requirements

*   **Node Name:** Requires a Fusion node named **`Control_Panel`** to exist in the current composition.
*   **Text Control:** Requires a Text Edit Control named **`SceneNumber`** on the node.
*   **Markers:** Requires one or more markers to be placed on the **Resolve Timeline Ruler** (not on individual clips).
*   **Execution:** This script must be run from the **Fusion Tab** within DaVinci Resolve.

## 💻 Installation

1.  **Save the Script:** Save the Python code below as a file named `Scene_Number_Stamper.py`.
2.  **Place the File:** Place the script in the Comp subfolder alongside your other Fusion-specific scripts.

    *   **Windows:** `%AppData%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\**Comp**`

## 🕹️ Usage

1.  **Open DaVinci Resolve Studio** and navigate to the **Fusion Page**.
2.  Ensure your Fusion Composition contains the required node named **`Control_Panel`** with the **`SceneNumber`** text control.
3.  Navigate to the **Workspace** menu.
4.  Select **Scripts** -> **Comp**.
5.  Click on **`Scene_Number_Stamper`**.


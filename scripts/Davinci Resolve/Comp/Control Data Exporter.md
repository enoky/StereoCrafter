

# DaVinci Resolve Script: Control Data Exporter (`Control_Data_Exporter.py`)

This script exports the values of configured Fusion Node controls into a structured JSON file. The data is sampled precisely at every frame position specified by a marker on the Resolve Timeline Ruler. This data can be used by external applications or other scripts (sidecars) for further processing.

## 🚀 Overview

The script's purpose is to act as a data bridge between your marked timeline events and external data processing tools.

**Output:** The script prompts the user for a save location and exports a single JSON file (saved with the extension `.fsexport` for safety) containing a structured list of markers, frame numbers, and the corresponding control values at those frames.

## 📋 Requirements

*   **Node Name:** Requires a Fusion node named **`Control_Panel`** to exist in the current composition, or if renamed then it is selected.
*   **Markers:** Requires one or more markers to be placed on the **Resolve Timeline Ruler** (not on individual clips).
*   **Execution:** This script must be run from the **Fusion Tab** within DaVinci Resolve (using the Workspace > Scripts menu).

### Configuration (Required)

To add or remove controls for data export, modify the `CONTROLS_TO_EXPORT` list at the top of the script.

| Item | Description | Example |
| :--- | :--- | :--- |
| **Control ID** | The scripting name of the control (e.g., `MaxDisparity`, `Convergence`). | `CONTROLS_TO_EXPORT = ["MaxDisparity", "Convergence"]` |

## 💻 Installation

1.  **Save the Script:** Save the Python code as a file named `Control_Data_Exporter.py`.
2.  **Place the File:** Place the script in the Comp subfolder. This ensures it appears in the Scripts menu when you are on the Fusion Page.

    *   **Windows:** `%AppData%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp`

## 🕹️ Usage

1.  **Open DaVinci Resolve Studio** and navigate to the **Fusion Page**.
2.  Ensure your Fusion Composition contains the required node named **`Control_Panel`** or is selected.
3.  Navigate to the **Workspace** menu.
4.  Select **Scripts** -> **Comp** (or your chosen subfolder).
5.  Click on **`Control_Data_Exporter`**.
6.  The script will execute and prompt you for a save location for the `.fsexport` JSON file.

***

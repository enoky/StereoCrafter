
# DaVinci Resolve Script: Markers to Keyframes (`Markers_To_Keyframes.py`)

This script automates the creation of identical keyframes on a defined set of Fusion Node controls at every position marked on the Resolve Timeline Ruler. This is ideal for synchronizing animation timing with edit points or client review markers.

## 🚀 Overview

The script serves to quickly apply animation timing dictated by timeline markers to a specific set of Fusion Node controls.

**⚠️ Destructive Operation:** This script explicitly **deletes all existing keyframes and animation splines** on the configured controls before inserting new ones.

## 📋 Requirements

*   **Node Name:** Requires a Fusion node named **`Control_Panel`** to exist in the current composition.
*   **Markers:** Requires one or more markers to be placed on the **Resolve Timeline Ruler** (not on individual clips).
*   **Execution:** This script must be run from the **Fusion Tab** within DaVinci Resolve (using the Workspace > Scripts menu).

### Configuration (Required)

To add or remove controls for keyframing, modify the `controls_to_keyframe` dictionary at the top of the script.

| Key | Description | Example |
| :--- | :--- | :--- |
| **Control ID** | The scripting name of the control on the `Control_Panel` node (e.g., `MaxDisparity`). | `"MaxDisparity"` |

The script supports two primary modes for setting values:

| Mode | Description | Example Config |
| :--- | :--- | :--- |
| `STATIC_VALUE` | Sets all new keyframes to the value the control currently holds. (Recommended for synchronization). | `"MaxDisparity": {"mode": "STATIC_VALUE"}` |
| `ALTERNATE_VALUES` | Alternates between two defined values for each successive marker. | `"FrontGain": {"mode": "ALTERNATE_VALUES", "value_low": 0.0, "value_high": 2.0}` |

## 💻 Installation

1.  **Save the Script:** Save the Python code as a file named `Markers_To_Keyframes.py`.
2.  **Place the File:** Place the script in the Comp subfolder. This ensures it appears in the Scripts menu when you are on the Fusion Page.

    *   **Windows:** `%AppData%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp`

## 🕹️ Usage

1.  **Open DaVinci Resolve Studio** and navigate to the **Fusion Page**.
2.  Ensure your Fusion Composition contains the required node named **`Control_Panel`**.
3.  Set the desired static value for any control configured for the `STATIC_VALUE` mode.
4.  Navigate to the **Workspace** menu.
5.  Select **Scripts** -> **Comp** (or your chosen subfolder).
6.  Click on **`Markers_To_Keyframes`**.
7.  The script will execute, delete existing keyframes, and insert new ones at every marker position.
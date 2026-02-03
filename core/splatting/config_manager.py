"""Configuration management for StereoCrafter applications.

Provides utilities for loading, saving, and managing application configuration
with support for defaults, backward compatibility, and file operations.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default configuration for Splatting GUI
SPLATTER_DEFAULT_CONFIG = {
    "DEFAULT_CONFIG_FILENAME": "config_splat.splatcfg",
    "input_source_clips": "./input_source_clips",
    "input_depth_maps": "./input_depth_maps",
    "multi_map_enabled": False,
    "output_splatted": "./output_splatted",
    "max_disp": "30.0",
    "process_length": "-1",
    "batch_size": "10",
    "dual_output": False,
    "enable_global_norm": False,
    "enable_full_resolution": True,
    "enable_low_resolution": False,
    "pre_res_width": "1920",
    "pre_res_height": "1080",
    "low_res_batch_size": "50",
    "convergence_point": "0.5",
    "output_crf": "23",
    "output_crf_full": "23",
    "output_crf_low": "23",
    "color_tags_mode": "Auto",
    "dark_mode_enabled": False,
    "skip_lowres_preproc": False,
    "move_to_finished": True,
    "crosshair_enabled": False,
    "crosshair_white": False,
    "crosshair_multi": False,
    "depth_pop_enabled": False,
    "auto_convergence_mode": "Off",
    "depth_gamma": "1.0",
    "depth_dilate_size_x": "3",
    "depth_dilate_size_y": "3",
    "depth_blur_size_x": "5",
    "depth_blur_size_y": "5",
    "depth_dilate_left": "0",
    "depth_blur_left": "0",
    "depth_blur_left_mix": "0.5",
    "enable_sidecar_gamma": True,
    "enable_sidecar_blur_dilate": True,
    "update_slider_from_sidecar": True,
    "auto_save_sidecar": False,
    "border_width": "0.0",
    "border_bias": "0.0",
    "border_mode": "Off",
    "auto_border_L": "0.0",
    "auto_border_R": "0.0",
    "preview_source": "Splat Result",
    "preview_size": "75%",
    "window_width": 620,
    "window_height": 750,
    "debug_mode_enabled": False,
    "border_manual": False,
}


def load_config(
    config_filename: str = "config_splat.splatcfg",
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load configuration from a JSON file with defaults.

    Args:
        config_filename: Path to the configuration file
        defaults: Dictionary of default values to use if file not found

    Returns:
        Dictionary containing configuration values
    """
    config = dict(defaults) if defaults else {}
    
    if not os.path.exists(config_filename):
        logger.debug(f"Config file not found: {config_filename}. Using defaults.")
        return config

    try:
        with open(config_filename, "r") as f:
            loaded_config = json.load(f)
        
        # Apply loaded values, preserving structure
        config.update(loaded_config)
        
        # Handle backward compatibility for specific keys
        _apply_backward_compat(config)
        
        logger.info(f"Loaded config from: {config_filename}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config file {config_filename}: {e}")
    except Exception as e:
        logger.error(f"Failed to load config file {config_filename}: {e}")

    return config


def save_config(
    config: Dict[str, Any],
    config_filename: str = "config_splat.splatcfg",
) -> bool:
    """Save configuration to a JSON file.

    Args:
        config: Dictionary of configuration values to save
        config_filename: Path to the configuration file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_filename, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved config to: {config_filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_filename}: {e}")
        return False


def load_settings_from_file(
    filename: str,
    tk_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load settings from a user-selected JSON file.

    Args:
        filename: Path to the settings file
        tk_vars: Optional dictionary mapping config keys to tkinter variables

    Returns:
        Dictionary of loaded settings
    """
    try:
        with open(filename, "r") as f:
            loaded_config = json.load(f)
        
        # Apply to tkinter variables if provided
        if tk_vars:
            for config_key, config_value in loaded_config.items():
                tk_var_attr_name = config_key + "_var"
                if tk_var_attr_name in tk_vars:
                    tk_var = tk_vars[tk_var_attr_name]
                    _set_tk_var(tk_var, config_value)
        
        logger.info(f"Loaded settings from: {filename}")
        return loaded_config
        
    except Exception as e:
        logger.error(f"Failed to load settings from {filename}: {e}")
        return {}


def save_settings_to_file(
    config: Dict[str, Any],
    filename: str,
) -> bool:
    """Save settings to a user-selected JSON file.

    Args:
        config: Dictionary of settings to save
        filename: Path to the output file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved settings to: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save settings to {filename}: {e}")
        return False


def get_current_config(
    tk_vars: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract current configuration from tkinter variables.

    Args:
        tk_vars: Dictionary mapping config keys to tkinter variables
        defaults: Optional dictionary of default values

    Returns:
        Dictionary of current configuration values
    """
    config = dict(defaults) if defaults else {}
    
    for key, var in tk_vars.items():
        if key.endswith("_var"):
            config_key = key[:-4]  # Remove "_var" suffix
            config[config_key] = _get_tk_var_value(var)
    
    return config


def reset_to_defaults(
    tk_vars: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    """Reset tkinter variables to their default values.

    Args:
        tk_vars: Dictionary mapping config keys to tkinter variables
        defaults: Dictionary of default values
    """
    for key, var in tk_vars.items():
        if key.endswith("_var"):
            config_key = key[:-4]
            default_value = defaults.get(config_key) if defaults else None
            if default_value is not None:
                _set_tk_var(var, default_value)


def _apply_backward_compat(config: Dict[str, Any]) -> None:
    """Apply backward compatibility transformations.

    Args:
        config: Configuration dictionary to modify
    """
    # Handle old 'enable_autogain' key
    # Old meaning: True = Raw Input / Disable Normalization (GN OFF)
    # New meaning: True = Enable Global Normalization (GN ON)
    if "enable_autogain" in config:
        old_value = config.pop("enable_autogain")
        config["enable_global_norm"] = not bool(old_value)

    # Handle old depth dilation erosion mapping (30..40 -> -0..-10)
    for key in ["depth_dilate_size_x", "depth_dilate_size_y"]:
        if key in config:
            try:
                val = float(config[key])
                if 30.0 < val <= 40.0:
                    config[key] = str(-(val - 30.0))
            except (ValueError, TypeError):
                pass


def _get_tk_var_value(var) -> Any:
    """Get the value from a tkinter variable.

    Args:
        var: Tkinter variable (StringVar, BooleanVar, IntVar, etc.)

    Returns:
        The value of the variable
    """
    import tkinter as tk
    
    if isinstance(var, tk.BooleanVar):
        return bool(var.get())
    elif isinstance(var, tk.IntVar):
        return int(var.get())
    elif isinstance(var, tk.DoubleVar):
        return float(var.get())
    else:
        return str(var.get())


def _set_tk_var(var, value: Any) -> None:
    """Set the value of a tkinter variable.

    Args:
        var: Tkinter variable (StringVar, BooleanVar, IntVar, etc.)
        value: Value to set
    """
    import tkinter as tk
    
    if isinstance(var, tk.BooleanVar):
        var.set(bool(value))
    elif isinstance(var, (tk.IntVar, tk.DoubleVar)):
        var.set(float(value))
    else:
        var.set(str(value))


class ConfigManager:
    """Manages application configuration with tkinter variable integration.

    Provides a high-level interface for loading, saving, and managing
    configuration with automatic synchronization to tkinter variables.

    Args:
        defaults: Optional dictionary of default configuration values
        config_filename: Name of the default config file
    """

    def __init__(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        config_filename: str = "config_splat.splatcfg",
    ):
        """Initialize the configuration manager.

        Args:
            defaults: Default configuration values
            config_filename: Name of the config file
        """
        self.defaults = defaults or SPLATTER_DEFAULT_CONFIG
        self.config_filename = config_filename
        self.config = {}
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Loaded configuration dictionary
        """
        self.config = load_config(self.config_filename, self.defaults)
        return self.config
    
    def save(self) -> bool:
        """Save current configuration to file.

        Returns:
            True if successful
        """
        return save_config(self.config, self.config_filename)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def sync_to_tk_vars(self, tk_vars: Dict[str, Any]) -> None:
        """Synchronize configuration to tkinter variables.

        Args:
            tk_vars: Dictionary mapping config keys to tkinter variables
        """
        for config_key, config_value in self.config.items():
            tk_var_attr_name = config_key + "_var"
            if tk_var_attr_name in tk_vars:
                _set_tk_var(tk_vars[tk_var_attr_name], config_value)
    
    def sync_from_tk_vars(self, tk_vars: Dict[str, Any]) -> None:
        """Synchronize configuration from tkinter variables.

        Args:
            tk_vars: Dictionary mapping config keys to tkinter variables
        """
        for key, var in tk_vars.items():
            if key.endswith("_var"):
                config_key = key[:-4]
                self.config[config_key] = _get_tk_var_value(var)

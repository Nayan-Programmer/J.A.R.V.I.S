"""
APP LAUNCHER
============
Opens desktop applications on the user's local PC using OS-native commands.
Works because the FastAPI backend runs locally on the user's machine.

Supports Windows, macOS, and Linux.
"""

import logging
import os
import subprocess
import sys
from typing import Tuple

logger = logging.getLogger("J.A.R.V.I.S")

# ── Protocol-handler apps (cross-platform) ─────────────────────────────────────
# These use URL protocol schemes registered by the installed desktop app.
PROTOCOL_MAP: dict[str, str] = {
    "whatsapp":  "whatsapp://",
    "spotify":   "spotify:",
    "discord":   "discord://",
    "telegram":  "tg://",
    "zoom":      "zoommtg://zoom.us/start",
    "slack":     "slack://",
    "teams":     "msteams://",
    "skype":     "skype:",
    "notion":    "notion://",
    "figma":     "figma://",
    "obsidian":  "obsidian://",
    "vscode":    "vscode://",
    "vs code":   "vscode://",
}

# ── Windows-specific executable commands ──────────────────────────────────────
WINDOWS_CMD_MAP: dict[str, str] = {
    "notepad":      "notepad",
    "calculator":   "calc",
    "calc":         "calc",
    "explorer":     "explorer",
    "file explorer":"explorer",
    "cmd":          "cmd",
    "terminal":     "cmd",
    "word":         "winword",
    "excel":        "excel",
    "powerpoint":   "powerpnt",
    "paint":        "mspaint",
    "task manager": "taskmgr",
    "control panel":"control",
    "settings":     "ms-settings:",
    "chrome":       "chrome",
    "firefox":      "firefox",
    "edge":         "msedge",
    "vlc":          "vlc",
    "snipping tool":"snippingtool",
    "snip":         "snippingtool",
}

# ── macOS app names (used with `open -a`) ─────────────────────────────────────
MACOS_APP_MAP: dict[str, str] = {
    "safari":       "Safari",
    "chrome":       "Google Chrome",
    "firefox":      "Firefox",
    "finder":       "Finder",
    "terminal":     "Terminal",
    "xcode":        "Xcode",
    "notes":        "Notes",
    "calculator":   "Calculator",
    "music":        "Music",
    "photos":       "Photos",
}


def launch_app(name: str) -> Tuple[bool, str]:
    """
    Try to open a desktop application by name.

    Strategy:
      1. Protocol-handler (works for WhatsApp, Spotify, Discord, etc.)
      2. Windows: shell command via subprocess
      3. macOS: `open -a <AppName>`
      4. Linux: `xdg-open` with known app names

    Returns:
        (success: bool, message: str)
    """
    key = (name or "").lower().strip()
    if not key:
        return False, "No app name provided"

    # ── 1. Protocol handler ───────────────────────────────────────────────────
    proto = PROTOCOL_MAP.get(key)
    if proto:
        try:
            if sys.platform == "win32":
                os.startfile(proto)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", proto])
            else:
                subprocess.Popen(["xdg-open", proto])
            logger.info("[LAUNCHER] Opened via protocol: %s → %s", key, proto)
            return True, f"Opening {name} app"
        except Exception as exc:
            logger.warning("[LAUNCHER] Protocol launch failed (%s → %s): %s", key, proto, exc)

    # ── 2. Windows executable ─────────────────────────────────────────────────
    if sys.platform == "win32":
        cmd = WINDOWS_CMD_MAP.get(key)
        if cmd:
            try:
                if cmd.endswith(":"):        # ms-settings: or similar
                    os.startfile(cmd)
                else:
                    subprocess.Popen(cmd, shell=True)
                logger.info("[LAUNCHER] Windows: launched %s via '%s'", key, cmd)
                return True, f"Opening {name}"
            except Exception as exc:
                logger.warning("[LAUNCHER] Windows cmd failed (%s): %s", cmd, exc)

        # Last resort: try ShellExecute with just the name
        try:
            os.startfile(key)
            return True, f"Opening {name}"
        except Exception:
            pass

    # ── 3. macOS `open -a` ────────────────────────────────────────────────────
    elif sys.platform == "darwin":
        app_name = MACOS_APP_MAP.get(key, name.title())
        try:
            subprocess.Popen(["open", "-a", app_name])
            logger.info("[LAUNCHER] macOS: open -a %s", app_name)
            return True, f"Opening {name}"
        except Exception as exc:
            logger.warning("[LAUNCHER] macOS open failed (%s): %s", app_name, exc)

    # ── 4. Linux xdg-open ─────────────────────────────────────────────────────
    else:
        try:
            subprocess.Popen(["xdg-open", key])
            return True, f"Opening {name}"
        except Exception as exc:
            logger.warning("[LAUNCHER] xdg-open failed (%s): %s", key, exc)

    return False, f"Could not find or launch '{name}' on this system"

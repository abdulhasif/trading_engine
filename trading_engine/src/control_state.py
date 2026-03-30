"""
src/control_state.py - Runtime Control for the Trading Engine
=============================================================
This file holds global flags that can be modified by the API or 
external scripts to control the engine's behavior without 
restarting the process.
"""

# The Global Control State dictionary
# Example: {"GLOBAL_KILL": True} will trigger immediate square-off.
CONTROL_STATE = {
    "GLOBAL_KILL": False,
    "PAUSE_ENTRY": False,
    "DEBUG_MODE": False
}

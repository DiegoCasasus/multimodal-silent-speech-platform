#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Fri 17 Jul 12:25:02 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from welcome_code
import socket
import time
import json
import subprocess
import shutil
from pylsl import StreamInfo, StreamOutlet
import atexit
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

labrecorder_proc = None
PREPARED_RUN = None
PREPARED_RUN_SOURCE = None


def resolve_project_path(value):
    """Resolve runtime paths relative to PROJECT_ROOT, never current cwd.

    Existing absolute paths continue to work. Future portable bridge files can
    store project-relative paths such as data/latest_prepared_run.json,
    recordings/..., brainprints/..., or code/decoder/...
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = os.path.expandvars(os.path.expanduser(text))
    if os.path.isabs(text):
        return os.path.normpath(text)

    return os.path.normpath(os.path.join(PROJECT_ROOT, text))


def load_project_config():
    """Load the optional project config used by the GUI/exportable version.

    This PsychoPy Builder script only relies on JSON to avoid requiring PyYAML
    inside the PsychoPy environment. Missing config is allowed; the experiment
    then falls back to the historical project-relative defaults.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, 'config', 'project_config.json'),
    ]

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                print(f"Loaded project config: {candidate}")
                return payload
            print(f"Ignoring project config with non-dict top level: {candidate}")
        except Exception as e:
            print(f"Could not load project config {candidate}: {e}")

    return {}


PROJECT_CONFIG = load_project_config()


def get_project_config_value(dotted_key, default=None):
    current = PROJECT_CONFIG
    for part in str(dotted_key).split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def get_configured_path(dotted_key, default=None):
    value = get_project_config_value(dotted_key, None)
    if value is None or str(value).strip() == '':
        value = default
    return resolve_project_path(value) if value else None


def get_decoder_python_exe():
    """Return the Python executable used for decoder/brainprint subprocesses."""
    configured = get_configured_path('external_tools.decoder_python_exe', None)
    if configured:
        if os.path.exists(configured):
            return configured
        print(f"Configured decoder_python_exe does not exist, falling back to sys.executable: {configured}")
    return sys.executable


def get_psychopy_python_exe():
    """Return the configured PsychoPy Python executable, if any.

    The current Builder-generated script normally runs inside the correct
    PsychoPy/Python process already, so this helper is mostly for future
    launchers and diagnostic output.
    """
    configured = get_configured_path('external_tools.psychopy_python_exe', None)
    if configured and os.path.exists(configured):
        return configured
    return sys.executable


BRIDGE_CANDIDATES = [
    get_configured_path('paths.bridge_file', os.path.join(PROJECT_ROOT, 'data', 'latest_prepared_run.json')),
    os.path.join(PROJECT_ROOT, 'data', 'latest_prepared_run.json'),
]
BRIDGE_CANDIDATES = [p for p in BRIDGE_CANDIDATES if p]

LABRECORDER_BASE_ROOT = get_configured_path(
    'paths.recordings_root',
    os.path.join(PROJECT_ROOT, 'recordings'),
)


def labrecorder_app_binary(app_path):
    """Return the executable inside a macOS .app bundle if it exists."""
    if not app_path:
        return None
    app_path = os.path.normpath(app_path)
    if not app_path.lower().endswith('.app') or not os.path.isdir(app_path):
        return None

    app_name = os.path.splitext(os.path.basename(app_path))[0]
    macos_dir = os.path.join(app_path, 'Contents', 'MacOS')
    candidates = [
        os.path.join(macos_dir, app_name),
        os.path.join(macos_dir, 'LabRecorder'),
        os.path.join(macos_dir, 'labrecorder'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def normalise_executable_candidate(path_value):
    if not path_value:
        return None
    path_value = resolve_project_path(path_value)
    if not path_value:
        return None

    app_binary = labrecorder_app_binary(path_value)
    if app_binary:
        return app_binary

    return path_value


LABRECORDER_EXE_CANDIDATES = [
    get_configured_path('external_tools.labrecorder_exe', None),
    os.path.join(SCRIPT_DIR, 'LabRecorder.exe'),
    os.path.join(SCRIPT_DIR, 'LabRecorder', 'LabRecorder.exe'),
    os.path.join(os.path.dirname(SCRIPT_DIR), 'LabRecorder.exe'),
    os.path.join(os.path.dirname(SCRIPT_DIR), 'LabRecorder', 'LabRecorder.exe'),
    os.path.join(PROJECT_ROOT, 'LabRecorder.exe'),
    os.path.join(PROJECT_ROOT, 'LabRecorder', 'LabRecorder.exe'),
    os.path.join(PROJECT_ROOT, 'LabRecorder.app'),
    os.path.join(PROJECT_ROOT, 'LabRecorder', 'LabRecorder.app'),
    os.path.join('/Applications', 'LabRecorder.app'),
    os.path.join('/Applications', 'LabRecorder.app', 'Contents', 'MacOS', 'LabRecorder'),
    '/opt/homebrew/opt/labrecorder/LabRecorder.app',
    '/opt/homebrew/opt/labrecorder/LabRecorder.app/Contents/MacOS/LabRecorder',
    '/usr/local/opt/labrecorder/LabRecorder.app',
    '/usr/local/opt/labrecorder/LabRecorder.app/Contents/MacOS/LabRecorder',
    r'C:\Program Files\LabRecorder\LabRecorder.exe',
    r'C:\Program Files (x86)\LabRecorder\LabRecorder.exe',
]


def resolve_prepared_run_paths(payload):
    """Normalise known bridge path fields to absolute paths for this script.

    This keeps the rest of the PsychoPy script stable: all existing code can
    continue using os.path.exists, os.makedirs, subprocess arguments, etc.
    """
    if not isinstance(payload, dict):
        return payload

    payload = dict(payload)

    for key in [
        'expected_xdf_path',
        'metadata_path',
        'manifest_path',
        'online_results_dir',
        'decoder_bundle_path',
        'brainprint_compare_json',
        'brainprint_summary_json',
    ]:
        if payload.get(key):
            payload[key] = resolve_project_path(payload[key])

    block_paths = payload.get('recording_block_paths')
    if isinstance(block_paths, dict):
        payload['recording_block_paths'] = {
            block: resolve_project_path(path_value) if path_value else path_value
            for block, path_value in block_paths.items()
        }

    ensemble = payload.get('decoder_ensemble')
    if isinstance(ensemble, dict):
        ensemble = dict(ensemble)
        components = []
        for component in ensemble.get('components', []):
            if isinstance(component, dict):
                component = dict(component)
                if component.get('bundle_path'):
                    component['bundle_path'] = resolve_project_path(component['bundle_path'])
            components.append(component)
        ensemble['components'] = components
        payload['decoder_ensemble'] = ensemble

    return payload


def force_window_focus(win):
    # On macOS, forcing activation of the pyglet/Cocoa window can freeze the next win.flip().
    # Keep the Windows/Linux behavior, but make this a no-op on macOS.
    import sys as _sys
    if _sys.platform == "darwin":
        print("[MAC] Skipping force_window_focus() to avoid Cocoa/pyglet flip freeze.", flush=True)
        return
    try:
        win.winHandle.activate()
        core.wait(0.15)
    except Exception as e:
        print(f"Could not force window focus: {e}", flush=True)


def load_prepared_run_bridge():
    global PREPARED_RUN_SOURCE

    for candidate in BRIDGE_CANDIDATES:
        if os.path.exists(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                payload = resolve_prepared_run_paths(payload)
                PREPARED_RUN_SOURCE = os.path.normpath(candidate)
                print(f"Loaded GUI bridge file: {candidate}")
                return payload
            except Exception as e:
                print(f"Could not read GUI bridge file at {candidate}: {e}")

    print("No GUI bridge file found.")
    return None


def apply_bridge_to_expinfo(expInfo, prepared_run):
    if not prepared_run:
        return expInfo

    expInfo['participant'] = prepared_run.get('participant_id', expInfo.get('participant', '001'))
    expInfo['session'] = prepared_run.get('session_id', expInfo.get('session', '001'))
    expInfo['run'] = prepared_run.get('run_id', expInfo.get('run', '001'))
    return expInfo


def resolve_block_name(block_num):
    if block_num == 1:
        return 'calibration'
    elif block_num == 2:
        return 'imagined'
    elif block_num == 3:
        return 'imagined'
    return f'block{block_num}'


def build_block_xdf_template(expInfo, block):
    participant = str(expInfo.get('participant', 'test_subject')).replace(' ', '_')
    session = str(expInfo.get('session', '001')).replace(' ', '_')
    run_id = str(expInfo.get('run', '001')).replace(' ', '_')

    if PREPARED_RUN is not None:
        experiment_mode = str(PREPARED_RUN.get('experiment_mode', 'recording')).replace(' ', '_')
    else:
        experiment_mode = 'recording'

    return f'exp_sub_{participant}_ses_{session}_run_{run_id}_mode_{experiment_mode}_bl_{block}.xdf'


def build_labrecorder_file_cmd(expInfo, block):
    """
    Priority:
    1. Use GUI-prepared per-block path if present.
    2. Fall back to GUI-prepared calibration folder / expected_xdf_path.
    3. Fall back to default recordings/<block>/ folder.
    """
    file_root = None
    template = None

    if PREPARED_RUN is not None:
        recording_block_paths = PREPARED_RUN.get('recording_block_paths', {})

        block_path = recording_block_paths.get(block)
        if block_path:
            block_path = os.path.normpath(block_path)
            file_root = os.path.dirname(block_path)
            template = os.path.basename(block_path)

        elif PREPARED_RUN.get('expected_xdf_path'):
            expected_xdf_path = os.path.normpath(PREPARED_RUN['expected_xdf_path'])
            file_root = os.path.dirname(expected_xdf_path)
            template = build_block_xdf_template(expInfo, block)

    if file_root is None:
        experiment_mode = PREPARED_RUN.get('experiment_mode', 'recording') if PREPARED_RUN else 'recording'
        file_root = os.path.join(LABRECORDER_BASE_ROOT, experiment_mode, block)
        template = build_block_xdf_template(expInfo, block)

    os.makedirs(file_root, exist_ok=True)
    file_root = os.path.normpath(file_root)
    if not file_root.endswith(os.sep):
        file_root = file_root + os.sep

    cmd = f'filename {{root:{file_root}}} {{template:{template}}}'
    print('Sending LabRecorder filename command:', cmd)
    return cmd


def labrecorder_is_listening(host='localhost', port=22345, timeout=0.5):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def executable_exists(path_value):
    candidate = normalise_executable_candidate(path_value)
    if not candidate or not os.path.exists(candidate):
        return None

    # Never select a bundled Windows executable on macOS/Linux merely because
    # the file exists inside the exported repository.
    if not sys.platform.startswith('win') and str(candidate).lower().endswith('.exe'):
        return None

    # On POSIX systems require an actually executable native binary.
    if not sys.platform.startswith('win') and not os.access(candidate, os.X_OK):
        return None

    return candidate


def find_labrecorder_exe():
    # 1) Project config override and explicit known candidates.
    for exe in LABRECORDER_EXE_CANDIDATES:
        candidate = executable_exists(exe)
        if candidate:
            return candidate

    # 2) PATH lookup.
    from_path = shutil.which('LabRecorder.exe') or shutil.which('LabRecorder') or shutil.which('labrecorder')
    if from_path:
        return from_path

    # 3) Shallow project search. Avoid deep recursive scans in large data folders.
    search_roots = [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR), PROJECT_ROOT]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth > 3:
                dirnames[:] = []
                continue

            for filename in ['LabRecorder.exe', 'LabRecorder', 'labrecorder']:
                if filename in filenames:
                    return os.path.join(dirpath, filename)

            for dirname in list(dirnames):
                if dirname.lower() == 'labrecorder.app':
                    app_binary = labrecorder_app_binary(os.path.join(dirpath, dirname))
                    if app_binary:
                        return app_binary

    return None


def ensure_labrecorder_running():
    global labrecorder_proc

    if labrecorder_is_listening():
        print('LabRecorder already running.')
        return True

    exe = find_labrecorder_exe()
    if exe is None:
        print('Could not find LabRecorder executable.')
        print('Checked explicit candidates:', LABRECORDER_EXE_CANDIDATES)
        print('Set external_tools.labrecorder_exe in config/project_config.json if needed.')
        return False

    try:
        print(f'Launching LabRecorder from: {exe}')
        labrecorder_proc = subprocess.Popen([exe])

        for _ in range(20):
            time.sleep(0.5)
            if labrecorder_is_listening():
                print('LabRecorder launched successfully and remote control is available.')
                return True

        print('LabRecorder launched, but remote-control port 22345 did not become available.')
    except Exception as e:
        print(f'Failed to launch LabRecorder from {exe}: {e}')

    return False
# Run 'Before Experiment' code from processing_code
online_calibration_proc = None
online_calibration_result = None
online_calibration_result_path = None
adaptation_allowed = False
adaptation_reason = ''
top_matches = []

selected_model_key = None
selected_mode_name = None

completed_online_trials = 0
predicted_word = ''
prediction_confidence = ''
feedback_text = ''
run_summary_text = ''

online_decoder_proc = None
online_decoder_log_handle = None
online_decoder_results_dir = None
online_decoder_service_status_path = None
online_decoder_latest_path = None
online_decoder_summary_path = None
online_decoder_launch_error = ''
online_decoder_ready = False

current_trial_index = 0
decoder_result_for_trial = None
expected_result_json_path = None
decode_wait_start_time = None
decode_timeout_sec = 10.0

processing_stage = ''
brainprint_candidate_npz = None
brainprint_compare_json = None
brainprint_best_match_file = ''
brainprint_best_match_cosine = None
brainprint_status = 'not_run'
processing_stage_started_at = None
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'Experiment_Online_Imagined'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'run': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath=__file__,
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=(0.0000, 0.0000, 0.0000), colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = (0.0000, 0.0000, 0.0000)
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    welcome_txt = visual.TextStim(win=win, name='welcome_txt',
        text='Welcome',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from welcome_code
    global PREPARED_RUN
    
    PREPARED_RUN = load_prepared_run_bridge()
    expInfo = apply_bridge_to_expinfo(expInfo, PREPARED_RUN)
    
    print("PREPARED_RUN =", PREPARED_RUN)
    print("expInfo after bridge =", expInfo)
    
    CONDITIONS_XLSX = get_configured_path("paths.conditions_xlsx", os.path.join(PROJECT_ROOT, "psychopy", "Conditions.xlsx"))
    
    if PREPARED_RUN is not None and PREPARED_RUN.get("basename"):
        thisExp.dataFileName = os.path.join(_thisDir, "data", PREPARED_RUN["basename"])
        print("Overriding PsychoPy data file name:", thisExp.dataFileName)
    
    # 1. Delayed fuse to prevent the final freeze
    atexit.register(lambda: os._exit(0))
    
    # 2. Set our starting block number
    block_num = 1
    imagined_batch = 1
    silent_batch = 1
    recording_started = False
    
    completed_online_trials = 0
    predicted_word = ''
    prediction_confidence = ''
    feedback_text = ''
    run_summary_text = ''
    online_decoder_proc = None
    online_decoder_results_dir = None
    current_trial_index = 0
    decoder_result_for_trial = None
    expected_result_json_path = None
    decode_wait_start_time = None
    decode_timeout_sec = 5.0
    # 3. Create the marker stream ONLY ONCE
    info = StreamInfo("PsychoPy_Markers", "Markers", 1, 0, "string", "my_imagined_speech_exp")
    outlet = StreamOutlet(info)
    time.sleep(0.1)
    print("LSL Marker Stream created for the whole experiment.")
    
    
    
    # --- Initialize components for Routine "cal_instructions" ---
    calStartKeys = keyboard.Keyboard(deviceName='defaultKeyboard')
    instructions = visual.TextStim(win=win, name='instructions',
        text='Calibration (no mouthing).\nLook at the center and COUNT how many NUMBERS you see.\nDo not press any keys during calibration.\nPress SPACE to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Start_record" ---
    
    # --- Initialize components for Routine "calibration" ---
    fixCal = visual.TextStim(win=win, name='fixCal',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    calStim = visual.TextStim(win=win, name='calStim',
        text='X',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.18, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Stop_record" ---
    
    # --- Initialize components for Routine "Processing_calibration" ---
    processing_text = visual.TextStim(win=win, name='processing_text',
        text='Processing calibration profile...\n\nPlease wait.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Instructions_imagined" ---
    instructions_key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    inst_imagined_txt = visual.TextStim(win=win, name='inst_imagined_txt',
        text='Look at the center cross.\n\nA white word will appear.\n\nWhen the word turns green, perform the imagined speech (imagine yourself saying the word without movement).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Start_record" ---
    
    # --- Initialize components for Routine "wait" ---
    ready_txt2 = visual.TextStim(win=win, name='ready_txt2',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "read" ---
    stimulus_txt_2 = visual.TextStim(win=win, name='stimulus_txt_2',
        text='',
        font='Arial',
        pos=(0, 0.08), draggable=False, height=0.17, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    phase_txt_read = visual.TextStim(win=win, name='phase_txt_read',
        text='Target cue',
        font='Arial',
        pos=(0, 0.28), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "trial" ---
    stimulus_go_txt = visual.TextStim(win=win, name='stimulus_go_txt',
        text='',
        font='Arial',
        pos=(0, 0.08), draggable=False, height=0.17, wrapWidth=None, ori=0.0, 
        color=(0.1294, 0.8667, 0.1294), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    phase_txt_trial = visual.TextStim(win=win, name='phase_txt_trial',
        text='Silently speak now',
        font='Arial',
        pos=(0, 0.28), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "decode_wait" ---
    decode_text = visual.TextStim(win=win, name='decode_text',
        text='Decoding...',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "online_feedback" ---
    feedback_txt = visual.TextStim(win=win, name='feedback_txt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "mini_break" ---
    break_key = keyboard.Keyboard(deviceName='defaultKeyboard')
    break_txt = visual.TextStim(win=win, name='break_txt',
        text='Rest break\\n\\nTake a short rest, blink, and swallow.\\n\\nPress SPACEBAR when you are ready to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Stop_record" ---
    
    # --- Initialize components for Routine "Finish" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[welcome_txt],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    thisExp.currentRoutine = Welcome
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_txt* updates
        
        # if welcome_txt is starting this frame...
        if welcome_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_txt.frameNStart = frameN  # exact frame index
            welcome_txt.tStart = t  # local t and not account for scr refresh
            welcome_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_txt.started')
            # update status
            welcome_txt.status = STARTED
            welcome_txt.setAutoDraw(True)
        
        # if welcome_txt is active this frame...
        if welcome_txt.status == STARTED:
            # update params
            pass
        
        # if welcome_txt is stopping this frame...
        if welcome_txt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > welcome_txt.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                welcome_txt.tStop = t  # not accounting for scr refresh
                welcome_txt.tStopRefresh = tThisFlipGlobal  # on global time
                welcome_txt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'welcome_txt.stopped')
                # update status
                welcome_txt.status = FINISHED
                welcome_txt.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Welcome,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Welcome.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Welcome.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Welcome.maxDurationReached:
        routineTimer.addTime(-Welcome.maxDuration)
    elif Welcome.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "cal_instructions" ---
    # create an object to store info about Routine cal_instructions
    cal_instructions = data.Routine(
        name='cal_instructions',
        components=[calStartKeys, instructions],
    )
    cal_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for calStartKeys
    calStartKeys.keys = []
    calStartKeys.rt = []
    _calStartKeys_allKeys = []
    # Run 'Begin Routine' code from calibration_code
    import random
    random.seed(12345)
    force_window_focus(win)
    win.callOnFlip(outlet.push_sample, ["CAL_BLOCK_START"])
    # store start times for cal_instructions
    cal_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    cal_instructions.tStart = globalClock.getTime(format='float')
    cal_instructions.status = STARTED
    thisExp.addData('cal_instructions.started', cal_instructions.tStart)
    cal_instructions.maxDuration = None
    # keep track of which components have finished
    cal_instructionsComponents = cal_instructions.components
    for thisComponent in cal_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "cal_instructions" ---
    thisExp.currentRoutine = cal_instructions
    cal_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *calStartKeys* updates
        waitOnFlip = False
        
        # if calStartKeys is starting this frame...
        if calStartKeys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calStartKeys.frameNStart = frameN  # exact frame index
            calStartKeys.tStart = t  # local t and not account for scr refresh
            calStartKeys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calStartKeys, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calStartKeys.started')
            # update status
            calStartKeys.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(calStartKeys.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(calStartKeys.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if calStartKeys.status == STARTED and not waitOnFlip:
            theseKeys = calStartKeys.getKeys(keyList=["space"], ignoreKeys=["escape"], waitRelease=False)
            _calStartKeys_allKeys.extend(theseKeys)
            if len(_calStartKeys_allKeys):
                calStartKeys.keys = _calStartKeys_allKeys[-1].name  # just the last key pressed
                calStartKeys.rt = _calStartKeys_allKeys[-1].rt
                calStartKeys.duration = _calStartKeys_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instructions* updates
        
        # if instructions is starting this frame...
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions.started')
            # update status
            instructions.status = STARTED
            instructions.setAutoDraw(True)
        
        # if instructions is active this frame...
        if instructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=cal_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            cal_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if cal_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in cal_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "cal_instructions" ---
    for thisComponent in cal_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for cal_instructions
    cal_instructions.tStop = globalClock.getTime(format='float')
    cal_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('cal_instructions.stopped', cal_instructions.tStop)
    # check responses
    if calStartKeys.keys in ['', [], None]:  # No response was made
        calStartKeys.keys = None
    thisExp.addData('calStartKeys.keys',calStartKeys.keys)
    if calStartKeys.keys != None:  # we had a response
        thisExp.addData('calStartKeys.rt', calStartKeys.rt)
        thisExp.addData('calStartKeys.duration', calStartKeys.duration)
    thisExp.nextEntry()
    # the Routine "cal_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Start_record" ---
    # create an object to store info about Routine Start_record
    Start_record = data.Routine(
        name='Start_record',
        components=[],
    )
    Start_record.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_start
    block = resolve_block_name(block_num)
    recording_started = False
    
    if not ensure_labrecorder_running():
        print("LabRecorder is not available. Recording was not started.")
    else:
        try:
            lr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            lr_socket.settimeout(2.0)
            lr_socket.connect(("localhost", 22345))
    
            def send_cmd(cmd):
                lr_socket.sendall(cmd.encode("utf-8") + b"\n")
                time.sleep(0.2)
                try:
                    lr_socket.recv(1024)
                except:
                    pass
    
            send_cmd("update")
            send_cmd("select all")
    
            file_cmd = build_labrecorder_file_cmd(expInfo, block)
            print("Sending LabRecorder filename command:", file_cmd)
            send_cmd(file_cmd)
    
            send_cmd("start")
            lr_socket.close()
    
            recording_started = True
            print(f"LabRecorder successfully started for BLOCK {block}!")
    
        except Exception as e:
            print("LabRecorder start failed:", e)
    
    # Advance experiment state even if LabRecorder failed,
    # otherwise downstream block logic breaks.
    block_num += 1
    # store start times for Start_record
    Start_record.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_record.tStart = globalClock.getTime(format='float')
    Start_record.status = STARTED
    thisExp.addData('Start_record.started', Start_record.tStart)
    Start_record.maxDuration = None
    # keep track of which components have finished
    Start_recordComponents = Start_record.components
    for thisComponent in Start_record.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_record" ---
    thisExp.currentRoutine = Start_record
    Start_record.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start_record,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start_record.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start_record.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start_record.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_record" ---
    for thisComponent in Start_record.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_record
    Start_record.tStop = globalClock.getTime(format='float')
    Start_record.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_record.stopped', Start_record.tStop)
    thisExp.nextEntry()
    # the Routine "Start_record" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    cal = data.TrialHandler2(
        name='cal',
        nReps=200.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(cal)  # add the loop to the experiment
    thisCal = cal.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCal.rgb)
    if thisCal != None:
        for paramName in thisCal:
            globals()[paramName] = thisCal[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisCal in cal:
        cal.status = STARTED
        if hasattr(thisCal, 'status'):
            thisCal.status = STARTED
        currentLoop = cal
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisCal.rgb)
        if thisCal != None:
            for paramName in thisCal:
                globals()[paramName] = thisCal[paramName]
        
        # --- Prepare to start Routine "calibration" ---
        # create an object to store info about Routine calibration
        calibration = data.Routine(
            name='calibration',
            components=[fixCal, calStim],
        )
        calibration.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from cal_code
        import random
        p_target = 0.10
        is_target = (random.random() < p_target)
        
        if is_target:
            cal_char = str(random.randint(0, 9))
            tag = "T"
        else:
            cal_char = chr(random.randint(ord('A'), ord('Z')))
            tag = "NT"
        
        # Set the stimulus text *here* (so it is defined before drawing)
        calStim.setText(cal_char)
        
        # Marker aligned to the screen flip
        win.callOnFlip(outlet.push_sample, [f"CAL_STIM_ON:{cal_char}:{tag}"])
        # store start times for calibration
        calibration.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        calibration.tStart = globalClock.getTime(format='float')
        calibration.status = STARTED
        thisExp.addData('calibration.started', calibration.tStart)
        calibration.maxDuration = None
        # keep track of which components have finished
        calibrationComponents = calibration.components
        for thisComponent in calibration.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "calibration" ---
        thisExp.currentRoutine = calibration
        calibration.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.6:
            # if trial has changed, end Routine now
            if hasattr(thisCal, 'status') and thisCal.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixCal* updates
            
            # if fixCal is starting this frame...
            if fixCal.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                fixCal.frameNStart = frameN  # exact frame index
                fixCal.tStart = t  # local t and not account for scr refresh
                fixCal.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixCal, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixCal.started')
                # update status
                fixCal.status = STARTED
                fixCal.setAutoDraw(True)
            
            # if fixCal is active this frame...
            if fixCal.status == STARTED:
                # update params
                pass
            
            # if fixCal is stopping this frame...
            if fixCal.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixCal.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    fixCal.tStop = t  # not accounting for scr refresh
                    fixCal.tStopRefresh = tThisFlipGlobal  # on global time
                    fixCal.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixCal.stopped')
                    # update status
                    fixCal.status = FINISHED
                    fixCal.setAutoDraw(False)
            
            # *calStim* updates
            
            # if calStim is starting this frame...
            if calStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                calStim.frameNStart = frameN  # exact frame index
                calStim.tStart = t  # local t and not account for scr refresh
                calStim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(calStim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'calStim.started')
                # update status
                calStim.status = STARTED
                calStim.setAutoDraw(True)
            
            # if calStim is active this frame...
            if calStim.status == STARTED:
                # update params
                pass
            
            # if calStim is stopping this frame...
            if calStim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > calStim.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    calStim.tStop = t  # not accounting for scr refresh
                    calStim.tStopRefresh = tThisFlipGlobal  # on global time
                    calStim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'calStim.stopped')
                    # update status
                    calStim.status = FINISHED
                    calStim.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=calibration,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                calibration.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if calibration.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in calibration.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "calibration" ---
        for thisComponent in calibration.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for calibration
        calibration.tStop = globalClock.getTime(format='float')
        calibration.tStopRefresh = tThisFlipGlobal
        thisExp.addData('calibration.stopped', calibration.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if calibration.maxDurationReached:
            routineTimer.addTime(-calibration.maxDuration)
        elif calibration.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.600000)
        # mark thisCal as finished
        if hasattr(thisCal, 'status'):
            thisCal.status = FINISHED
        # if awaiting a pause, pause now
        if cal.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            cal.status = STARTED
        thisExp.nextEntry()
        
    # completed 200.0 repeats of 'cal'
    cal.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Stop_record" ---
    # create an object to store info about Routine Stop_record
    Stop_record = data.Routine(
        name='Stop_record',
        components=[],
    )
    Stop_record.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from stop_code
    win.callOnFlip(outlet.push_sample, [f"{block}_BLOCK_END"])
    print(f"{block} block finished")
    # store start times for Stop_record
    Stop_record.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Stop_record.tStart = globalClock.getTime(format='float')
    Stop_record.status = STARTED
    thisExp.addData('Stop_record.started', Stop_record.tStart)
    Stop_record.maxDuration = None
    # keep track of which components have finished
    Stop_recordComponents = Stop_record.components
    for thisComponent in Stop_record.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Stop_record" ---
    thisExp.currentRoutine = Stop_record
    Stop_record.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Stop_record,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Stop_record.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Stop_record.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Stop_record.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Stop_record" ---
    for thisComponent in Stop_record.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Stop_record
    Stop_record.tStop = globalClock.getTime(format='float')
    Stop_record.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Stop_record.stopped', Stop_record.tStop)
    # Run 'End Routine' code from stop_code
    try:
        stop_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stop_socket.settimeout(2.0)
        stop_socket.connect(("localhost", 22345))
        stop_socket.sendall(b"stop\n")
        stop_socket.close()
        print("Loop finished. LabRecorder stopped and block file saved.")
    except:
        pass
    thisExp.nextEntry()
    # the Routine "Stop_record" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Processing_calibration" ---
    # create an object to store info about Routine Processing_calibration
    Processing_calibration = data.Routine(
        name='Processing_calibration',
        components=[processing_text],
    )
    Processing_calibration.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from processing_code
    continueRoutine = True
    
    online_calibration_result = None
    adaptation_allowed = False
    adaptation_reason = ''
    top_matches = []
    
    selected_model_key = None if PREPARED_RUN is None else PREPARED_RUN.get('selected_model')
    selected_mode_name = None if PREPARED_RUN is None else PREPARED_RUN.get('selected_mode')
    
    online_decoder_launch_error = ''
    online_decoder_ready = False
    
    current_trial_index = 0
    decoder_result_for_trial = None
    expected_result_json_path = None
    decode_wait_start_time = None
    decode_timeout_sec = 10.0
    
    brainprint_candidate_npz = None
    brainprint_compare_json = None
    brainprint_best_match_file = ''
    brainprint_best_match_cosine = None
    brainprint_status = 'not_run'
    processing_stage_started_at = globalClock.getTime()
    
    if PREPARED_RUN is None:
        processing_text.text = "No prepared run bridge found."
        adaptation_reason = "No prepared run bridge found."
        processing_stage = "failed"
    
    else:
        online_decoder_results_dir = PREPARED_RUN.get('online_results_dir')
        if online_decoder_results_dir:
            os.makedirs(online_decoder_results_dir, exist_ok=True)
    
        online_decoder_service_status_path = (
            None if not online_decoder_results_dir
            else os.path.join(online_decoder_results_dir, 'service_status.json')
        )
        online_decoder_latest_path = (
            None if not online_decoder_results_dir
            else os.path.join(online_decoder_results_dir, 'latest.json')
        )
        online_decoder_summary_path = (
            None if not online_decoder_results_dir
            else os.path.join(online_decoder_results_dir, 'run_summary.json')
        )
    
        # clean stale files for this run
        if online_decoder_results_dir and os.path.isdir(online_decoder_results_dir):
            for stale_name in ['service_status.json', 'latest.json', 'run_summary.json']:
                stale_path = os.path.join(online_decoder_results_dir, stale_name)
                if os.path.exists(stale_path):
                    try:
                        os.remove(stale_path)
                    except Exception:
                        pass
    
            for filename in os.listdir(online_decoder_results_dir):
                if filename.startswith('trial_') and filename.endswith('.json'):
                    try:
                        os.remove(os.path.join(online_decoder_results_dir, filename))
                    except Exception:
                        pass
    
        processing_text.text = "Processing calibration...\n\nComputing brainprint."
        processing_stage = "brainprint_compute_start"
    # store start times for Processing_calibration
    Processing_calibration.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Processing_calibration.tStart = globalClock.getTime(format='float')
    Processing_calibration.status = STARTED
    thisExp.addData('Processing_calibration.started', Processing_calibration.tStart)
    Processing_calibration.maxDuration = None
    # keep track of which components have finished
    Processing_calibrationComponents = Processing_calibration.components
    for thisComponent in Processing_calibration.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Processing_calibration" ---
    thisExp.currentRoutine = Processing_calibration
    Processing_calibration.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from processing_code
        if processing_stage == "failed": 
            continueRoutine = False 
         
        elif processing_stage == "brainprint_compute_start": 
            calibration_xdf = None 
            if PREPARED_RUN is not None: 
                calibration_xdf = PREPARED_RUN.get('recording_block_paths', {}).get('calibration') 
         
            has_eeg = selected_mode_name is not None and ('EEG' in str(selected_mode_name)) 
         
            if (not has_eeg) or (not calibration_xdf) or (not os.path.exists(calibration_xdf)): 
                brainprint_status = 'skipped' 
                adaptation_reason = 'Skipping brainprint: no EEG calibration file available.' 
                processing_text.text = "Processing calibration...\n\nSkipping brainprint.\nLaunching decoder." 
                processing_stage = "decoder_start" 
         
            else: 
                brainprint_candidate_npz = os.path.join( 
                    PROJECT_ROOT, 
                    'brainprints', 
                    'candidates', 
                    os.path.splitext(os.path.basename(calibration_xdf))[0] + '.npz' 
                ) 
         
                compute_script = os.path.join(PROJECT_ROOT, 'code', 'decoder', 'compute_brainprint.py') 
                compute_cmd = [ 
                    get_decoder_python_exe(), 
                    compute_script, 
                    '--xdf', calibration_xdf, 
                    '--out', brainprint_candidate_npz, 
                    '--signal-stream-type', 'EEG', 
                    '--marker-stream-name', 'PsychoPy_Markers', 
                ] 
         
                print("Running brainprint compute:", " ".join(compute_cmd)) 
                processing_text.text = "Processing calibration...\n\nComputing brainprint." 
         
                try: 
                    compute_res = subprocess.run( 
                        compute_cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=60, 
                    ) 
         
                    if compute_res.stdout: 
                        print("Brainprint compute stdout:\n", compute_res.stdout) 
                    if compute_res.stderr: 
                        print("Brainprint compute stderr:\n", compute_res.stderr) 
         
                    bank_dir = get_configured_path('paths.brainprint_bank_dir', os.path.join(PROJECT_ROOT, 'brainprints', 'bank')) 
                    bank_has_npz = False 
                    if os.path.isdir(bank_dir): 
                        for fname in os.listdir(bank_dir): 
                            if fname.endswith('.npz'): 
                                bank_has_npz = True 
                                break 
         
                    if compute_res.returncode == 0 and brainprint_candidate_npz and os.path.exists(brainprint_candidate_npz) and bank_has_npz: 
                        processing_text.text = "Processing calibration...\n\nComparing brainprint against bank." 
                        processing_stage = "brainprint_compare_start" 
                    elif compute_res.returncode == 0: 
                        brainprint_status = 'ok_no_bank' 
                        adaptation_reason = 'Brainprint computed, but bank comparison skipped because bank is empty.' 
                        processing_text.text = "Processing calibration...\n\nBrainprint computed.\nNo bank available.\nLaunching decoder." 
                        processing_stage = "decoder_start" 
                    else: 
                        brainprint_status = 'failed' 
                        adaptation_allowed = False 
                        adaptation_reason = 'Brainprint computation failed.' 
                        processing_text.text = "Processing calibration...\n\nBrainprint computation failed.\nLaunching decoder without adaptation." 
                        processing_stage = "decoder_start" 
         
                except subprocess.TimeoutExpired: 
                    brainprint_status = 'failed' 
                    adaptation_allowed = False 
                    adaptation_reason = 'Brainprint computation timed out.' 
                    processing_text.text = "Processing calibration...\n\nBrainprint computation timed out.\nLaunching decoder without adaptation." 
                    processing_stage = "decoder_start" 
         
        elif processing_stage == "brainprint_compute_wait": 
            if brainprint_compute_proc is not None: 
                return_code = brainprint_compute_proc.poll() 
                if return_code is not None: 
                    stdout_text, stderr_text = brainprint_compute_proc.communicate() 
         
                    if stdout_text: 
                        print("Brainprint compute stdout:\n", stdout_text) 
                    if stderr_text: 
                        print("Brainprint compute stderr:\n", stderr_text) 
         
                    bank_dir = get_configured_path('paths.brainprint_bank_dir', os.path.join(PROJECT_ROOT, 'brainprints', 'bank')) 
                    bank_has_npz = False 
                    if os.path.isdir(bank_dir): 
                        for fname in os.listdir(bank_dir): 
                            if fname.endswith('.npz'): 
                                bank_has_npz = True 
                                break 
         
                    if return_code == 0 and brainprint_candidate_npz and os.path.exists(brainprint_candidate_npz) and bank_has_npz: 
                        processing_text.text = "Processing calibration...\n\nComparing brainprint against bank." 
                        processing_stage = "brainprint_compare_start" 
                    elif return_code == 0: 
                        brainprint_status = 'ok_no_bank' 
                        adaptation_reason = 'Brainprint computed, but bank comparison skipped because bank is empty.' 
                        processing_text.text = "Processing calibration...\n\nBrainprint computed.\nNo bank available.\nLaunching decoder." 
                        processing_stage = "decoder_start" 
                    else: 
                        brainprint_status = 'failed' 
                        adaptation_reason = 'Brainprint computation failed.' 
                        processing_text.text = "Processing calibration...\n\nBrainprint computation failed.\nLaunching decoder without adaptation." 
                        processing_stage = "decoder_start" 
         
        elif processing_stage == "brainprint_compare_start": 
            compare_script = os.path.join(PROJECT_ROOT, 'code', 'decoder', 'compare_brainprint.py') 
            brainprint_compare_json = os.path.join( 
                PROJECT_ROOT, 
                'brainprints', 
                'comparisons', 
                os.path.splitext(os.path.basename(brainprint_candidate_npz))[0] + '_compare.json' 
            ) 
         
            compare_cmd = [ 
                get_decoder_python_exe(), 
                compare_script, 
                '--mode', 'query', 
                '--query', brainprint_candidate_npz, 
                '--bank-dir', get_configured_path('paths.brainprint_bank_dir', os.path.join(PROJECT_ROOT, 'brainprints', 'bank')), 
                '--top-k', '10', 
                '--strict-meta-keys', 'signal_type', 'fs', 'tmin', 'tmax', 
                '--out', brainprint_compare_json, 
            ] 
         
            print("Running brainprint compare:", " ".join(compare_cmd)) 
            processing_text.text = "Processing calibration...\n\nComparing brainprint against bank." 
         
            try: 
                compare_res = subprocess.run( 
                    compare_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=60, 
                ) 
         
                if compare_res.stdout: 
                    print("Brainprint compare stdout:\n", compare_res.stdout) 
                if compare_res.stderr: 
                    print("Brainprint compare stderr:\n", compare_res.stderr) 
         
                if compare_res.returncode == 0 and os.path.exists(brainprint_compare_json): 
                    try: 
                        with open(brainprint_compare_json, 'r', encoding='utf-8') as f: 
                            compare_payload = json.load(f) 
                        top_matches = compare_payload.get('ranked_results', []) 
                    except Exception: 
                        top_matches = [] 
         
                    if len(top_matches) > 0: 
                        brainprint_best_match_file = top_matches[0].get('file', '') 
                        brainprint_best_match_cosine = top_matches[0].get('cosine_similarity', None) 
                        adaptation_allowed = True 
                        adaptation_reason = 'Brainprint compared successfully.' 
                    else: 
                        adaptation_allowed = False 
                        adaptation_reason = 'Brainprint compare produced no ranked results.' 
         
                    brainprint_status = 'ok' 
                    processing_text.text = "Processing calibration...\n\nBrainprint compared.\nLaunching decoder." 
                else: 
                    brainprint_status = 'failed' 
                    adaptation_allowed = False 
                    adaptation_reason = 'Brainprint comparison failed.' 
                    processing_text.text = "Processing calibration...\n\nBrainprint comparison failed.\nLaunching decoder without adaptation." 
         
                processing_stage = "decoder_start" 
         
            except subprocess.TimeoutExpired: 
                brainprint_status = 'failed' 
                adaptation_allowed = False 
                adaptation_reason = 'Brainprint comparison timed out.' 
                processing_text.text = "Processing calibration...\n\nBrainprint comparison timed out.\nLaunching decoder without adaptation." 
                processing_stage = "decoder_start" 
         
        elif processing_stage == "brainprint_compare_wait": 
            if brainprint_compare_proc is not None: 
                return_code = brainprint_compare_proc.poll() 
                if return_code is not None: 
                    stdout_text, stderr_text = brainprint_compare_proc.communicate() 
         
                    if stdout_text: 
                        print("Brainprint compare stdout:\n", stdout_text) 
                    if stderr_text: 
                        print("Brainprint compare stderr:\n", stderr_text) 
         
                    if return_code == 0 and brainprint_compare_json and os.path.exists(brainprint_compare_json): 
                        try: 
                            with open(brainprint_compare_json, 'r', encoding='utf-8') as f: 
                                compare_payload = json.load(f) 
                            top_matches = compare_payload.get('ranked_results', []) 
                        except Exception: 
                            top_matches = [] 
         
                        if len(top_matches) > 0: 
                            brainprint_best_match_file = top_matches[0].get('file', '') 
                            brainprint_best_match_cosine = top_matches[0].get('cosine_similarity', None) 
                            adaptation_allowed = True 
                            adaptation_reason = 'Brainprint compared successfully.' 
                        else: 
                            adaptation_allowed = False 
                            adaptation_reason = 'Brainprint compare produced no ranked results.' 
         
                        brainprint_status = 'ok' 
                        processing_text.text = "Processing calibration...\n\nBrainprint compared.\nLaunching decoder." 
                    else: 
                        brainprint_status = 'failed' 
                        adaptation_allowed = False 
                        adaptation_reason = 'Brainprint comparison failed.' 
                        processing_text.text = "Processing calibration...\n\nBrainprint comparison failed.\nLaunching decoder without adaptation." 
         
                    processing_stage = "decoder_start" 
         
        elif processing_stage == "decoder_start": 
            if PREPARED_RUN is None: 
                online_decoder_launch_error = 'PREPARED_RUN is missing.' 
                processing_text.text = "Could not launch decoder:\nPREPARED_RUN missing." 
                processing_stage = "decoder_failed" 
         
            else: 
                decoder_script = get_configured_path('paths.online_decoder_service_script', os.path.join(PROJECT_ROOT, 'code', 'decoder', 'online_decoder_service.py')) 
                bridge_path = PREPARED_RUN_SOURCE if PREPARED_RUN_SOURCE else os.path.join(PROJECT_ROOT, 'data', 'latest_prepared_run.json') 
                bundle_path = PREPARED_RUN.get('decoder_bundle_path') 
                summary_json = PREPARED_RUN.get('brainprint_summary_json') 
         
                cmd = [ 
                    get_decoder_python_exe(), 
                    decoder_script, 
                    '--bridge', bridge_path, 
                    '--results-dir', online_decoder_results_dir, 
                    '--marker-stream-name', 'PsychoPy_Markers', 
                    '--mode', PREPARED_RUN.get('experiment_mode', 'online_imagined'), 
                    '--device', 'cpu', 
                ] 
         
                if bundle_path: 
                    cmd.extend(['--bundle-path', bundle_path]) 
         
                if brainprint_compare_json and os.path.exists(brainprint_compare_json): 
                    cmd.extend(['--brainprint-compare-json', brainprint_compare_json]) 
         
                if summary_json and os.path.exists(summary_json): 
                    cmd.extend(['--brainprint-summary-json', summary_json]) 
         
                print("Launching online decoder service:") 
                print(" ".join(cmd)) 
         
                online_decoder_proc = subprocess.Popen(cmd) 
                processing_text.text = "Processing calibration...\n\nLaunching decoder service." 
                processing_stage_started_at = globalClock.getTime() 
                processing_stage = "decoder_wait_ready" 
         
        elif processing_stage == "decoder_wait_ready": 
            status_payload = None 
         
            if online_decoder_service_status_path and os.path.exists(online_decoder_service_status_path): 
                try: 
                    with open(online_decoder_service_status_path, 'r', encoding='utf-8') as f: 
                        status_payload = json.load(f) 
                except Exception: 
                    status_payload = None 
         
            if status_payload is not None: 
                decoder_status = status_payload.get('status', 'unknown') 
                processing_text.text = f"Processing calibration...\n\nDecoder status: {decoder_status}" 
         
                if decoder_status == 'running': 
                    online_decoder_ready = True 
                    continueRoutine = False 
         
            if online_decoder_proc is not None and online_decoder_proc.poll() is not None and not online_decoder_ready: 
                online_decoder_launch_error = 'Online decoder service exited before reaching running state.' 
                processing_text.text = online_decoder_launch_error 
                processing_stage = "decoder_failed" 
         
            elif globalClock.getTime() - processing_stage_started_at >= 20.0 and not online_decoder_ready: 
                online_decoder_launch_error = 'Timed out waiting for decoder service to reach running state.' 
                processing_text.text = online_decoder_launch_error 
                processing_stage = "decoder_failed" 
         
        elif processing_stage == "decoder_failed": 
            adaptation_allowed = False 
            if not adaptation_reason: 
                adaptation_reason = online_decoder_launch_error if online_decoder_launch_error else 'Decoder launch failed.' 
            continueRoutine = False
        
        # *processing_text* updates
        
        # if processing_text is starting this frame...
        if processing_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            processing_text.frameNStart = frameN  # exact frame index
            processing_text.tStart = t  # local t and not account for scr refresh
            processing_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(processing_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'processing_text.started')
            # update status
            processing_text.status = STARTED
            processing_text.setAutoDraw(True)
        
        # if processing_text is active this frame...
        if processing_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Processing_calibration,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Processing_calibration.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Processing_calibration.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Processing_calibration.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Processing_calibration" ---
    for thisComponent in Processing_calibration.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Processing_calibration
    Processing_calibration.tStop = globalClock.getTime(format='float')
    Processing_calibration.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Processing_calibration.stopped', Processing_calibration.tStop)
    # Run 'End Routine' code from processing_code
    thisExp.addData('adaptation_allowed', adaptation_allowed)
    thisExp.addData('adaptation_reason', adaptation_reason)
    thisExp.addData('brainprint_status', brainprint_status)
    thisExp.addData('brainprint_candidate_npz', brainprint_candidate_npz if brainprint_candidate_npz else '')
    thisExp.addData('brainprint_compare_json', brainprint_compare_json if brainprint_compare_json else '')
    thisExp.addData('online_decoder_results_dir', online_decoder_results_dir if online_decoder_results_dir else '')
    thisExp.addData('online_decoder_service_status_path', online_decoder_service_status_path if online_decoder_service_status_path else '')
    thisExp.addData('online_decoder_ready', online_decoder_ready)
    
    if brainprint_best_match_file:
        thisExp.addData('best_match_file', brainprint_best_match_file)
    else:
        thisExp.addData('best_match_file', '')
    
    if brainprint_best_match_cosine is not None:
        thisExp.addData('best_match_cosine', brainprint_best_match_cosine)
    else:
        thisExp.addData('best_match_cosine', '')
    thisExp.nextEntry()
    # the Routine "Processing_calibration" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instructions_imagined" ---
    # create an object to store info about Routine Instructions_imagined
    Instructions_imagined = data.Routine(
        name='Instructions_imagined',
        components=[instructions_key_resp_2, inst_imagined_txt],
    )
    Instructions_imagined.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instructions_key_resp_2
    instructions_key_resp_2.keys = []
    instructions_key_resp_2.rt = []
    _instructions_key_resp_2_allKeys = []
    # Run 'Begin Routine' code from code_instructions_imagined
    force_window_focus(win)
    # store start times for Instructions_imagined
    Instructions_imagined.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions_imagined.tStart = globalClock.getTime(format='float')
    Instructions_imagined.status = STARTED
    thisExp.addData('Instructions_imagined.started', Instructions_imagined.tStart)
    Instructions_imagined.maxDuration = None
    # keep track of which components have finished
    Instructions_imaginedComponents = Instructions_imagined.components
    for thisComponent in Instructions_imagined.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions_imagined" ---
    thisExp.currentRoutine = Instructions_imagined
    Instructions_imagined.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_key_resp_2* updates
        waitOnFlip = False
        
        # if instructions_key_resp_2 is starting this frame...
        if instructions_key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_key_resp_2.frameNStart = frameN  # exact frame index
            instructions_key_resp_2.tStart = t  # local t and not account for scr refresh
            instructions_key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_key_resp_2.started')
            # update status
            instructions_key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instructions_key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instructions_key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instructions_key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = instructions_key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instructions_key_resp_2_allKeys.extend(theseKeys)
            if len(_instructions_key_resp_2_allKeys):
                instructions_key_resp_2.keys = _instructions_key_resp_2_allKeys[-1].name  # just the last key pressed
                instructions_key_resp_2.rt = _instructions_key_resp_2_allKeys[-1].rt
                instructions_key_resp_2.duration = _instructions_key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *inst_imagined_txt* updates
        
        # if inst_imagined_txt is starting this frame...
        if inst_imagined_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            inst_imagined_txt.frameNStart = frameN  # exact frame index
            inst_imagined_txt.tStart = t  # local t and not account for scr refresh
            inst_imagined_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(inst_imagined_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'inst_imagined_txt.started')
            # update status
            inst_imagined_txt.status = STARTED
            inst_imagined_txt.setAutoDraw(True)
        
        # if inst_imagined_txt is active this frame...
        if inst_imagined_txt.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Instructions_imagined,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instructions_imagined.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instructions_imagined.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instructions_imagined.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions_imagined" ---
    for thisComponent in Instructions_imagined.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions_imagined
    Instructions_imagined.tStop = globalClock.getTime(format='float')
    Instructions_imagined.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions_imagined.stopped', Instructions_imagined.tStop)
    # check responses
    if instructions_key_resp_2.keys in ['', [], None]:  # No response was made
        instructions_key_resp_2.keys = None
    thisExp.addData('instructions_key_resp_2.keys',instructions_key_resp_2.keys)
    if instructions_key_resp_2.keys != None:  # we had a response
        thisExp.addData('instructions_key_resp_2.rt', instructions_key_resp_2.rt)
        thisExp.addData('instructions_key_resp_2.duration', instructions_key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Instructions_imagined" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Start_record" ---
    # create an object to store info about Routine Start_record
    Start_record = data.Routine(
        name='Start_record',
        components=[],
    )
    Start_record.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_start
    block = resolve_block_name(block_num)
    recording_started = False
    
    if not ensure_labrecorder_running():
        print("LabRecorder is not available. Recording was not started.")
    else:
        try:
            lr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            lr_socket.settimeout(2.0)
            lr_socket.connect(("localhost", 22345))
    
            def send_cmd(cmd):
                lr_socket.sendall(cmd.encode("utf-8") + b"\n")
                time.sleep(0.2)
                try:
                    lr_socket.recv(1024)
                except:
                    pass
    
            send_cmd("update")
            send_cmd("select all")
    
            file_cmd = build_labrecorder_file_cmd(expInfo, block)
            print("Sending LabRecorder filename command:", file_cmd)
            send_cmd(file_cmd)
    
            send_cmd("start")
            lr_socket.close()
    
            recording_started = True
            print(f"LabRecorder successfully started for BLOCK {block}!")
    
        except Exception as e:
            print("LabRecorder start failed:", e)
    
    # Advance experiment state even if LabRecorder failed,
    # otherwise downstream block logic breaks.
    block_num += 1
    # store start times for Start_record
    Start_record.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_record.tStart = globalClock.getTime(format='float')
    Start_record.status = STARTED
    thisExp.addData('Start_record.started', Start_record.tStart)
    Start_record.maxDuration = None
    # keep track of which components have finished
    Start_recordComponents = Start_record.components
    for thisComponent in Start_record.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_record" ---
    thisExp.currentRoutine = Start_record
    Start_record.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start_record,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start_record.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start_record.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start_record.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_record" ---
    for thisComponent in Start_record.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_record
    Start_record.tStop = globalClock.getTime(format='float')
    Start_record.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_record.stopped', Start_record.tStop)
    thisExp.nextEntry()
    # the Routine "Start_record" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_online_imagined = data.TrialHandler2(
        name='trials_online_imagined',
        nReps=45.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials_online_imagined)  # add the loop to the experiment
    thisTrials_online_imagined = trials_online_imagined.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_online_imagined.rgb)
    if thisTrials_online_imagined != None:
        for paramName in thisTrials_online_imagined:
            globals()[paramName] = thisTrials_online_imagined[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_online_imagined in trials_online_imagined:
        trials_online_imagined.status = STARTED
        if hasattr(thisTrials_online_imagined, 'status'):
            thisTrials_online_imagined.status = STARTED
        currentLoop = trials_online_imagined
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_online_imagined.rgb)
        if thisTrials_online_imagined != None:
            for paramName in thisTrials_online_imagined:
                globals()[paramName] = thisTrials_online_imagined[paramName]
        
        # --- Prepare to start Routine "wait" ---
        # create an object to store info about Routine wait
        wait = data.Routine(
            name='wait',
            components=[ready_txt2],
        )
        wait.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        import numpy as np
        iti = np.random.uniform(0.5, 1.5)
        # store start times for wait
        wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wait.tStart = globalClock.getTime(format='float')
        wait.status = STARTED
        thisExp.addData('wait.started', wait.tStart)
        wait.maxDuration = None
        # keep track of which components have finished
        waitComponents = wait.components
        for thisComponent in wait.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "wait" ---
        thisExp.currentRoutine = wait
        wait.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_online_imagined, 'status') and thisTrials_online_imagined.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_2
            if t >= iti: 
                continueRoutine = False
            
            # *ready_txt2* updates
            
            # if ready_txt2 is starting this frame...
            if ready_txt2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                ready_txt2.frameNStart = frameN  # exact frame index
                ready_txt2.tStart = t  # local t and not account for scr refresh
                ready_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ready_txt2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ready_txt2.started')
                # update status
                ready_txt2.status = STARTED
                ready_txt2.setAutoDraw(True)
            
            # if ready_txt2 is active this frame...
            if ready_txt2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=wait,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                wait.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if wait.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in wait.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait" ---
        for thisComponent in wait.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wait
        wait.tStop = globalClock.getTime(format='float')
        wait.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wait.stopped', wait.tStop)
        # the Routine "wait" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "read" ---
        # create an object to store info about Routine read
        read = data.Routine(
            name='read',
            components=[stimulus_txt_2, phase_txt_read],
        )
        read.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from read_code
        cue_duration = 1.5
        feedback_text = ""
        predicted_word = ""
        prediction_confidence = ""
        
        win.callOnFlip(outlet.push_sample, [f"IMAGINED_CUE:{target_word}"])
        stimulus_txt_2.setText(target_word)
        stimulus_txt_2.setText(target_word)
        # store start times for read
        read.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        read.tStart = globalClock.getTime(format='float')
        read.status = STARTED
        thisExp.addData('read.started', read.tStart)
        read.maxDuration = None
        # keep track of which components have finished
        readComponents = read.components
        for thisComponent in read.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "read" ---
        thisExp.currentRoutine = read
        read.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_online_imagined, 'status') and thisTrials_online_imagined.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from read_code
            if t >= cue_duration: 
                continueRoutine = False
            
            # *stimulus_txt_2* updates
            
            # if stimulus_txt_2 is starting this frame...
            if stimulus_txt_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                stimulus_txt_2.frameNStart = frameN  # exact frame index
                stimulus_txt_2.tStart = t  # local t and not account for scr refresh
                stimulus_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stimulus_txt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stimulus_txt_2.started')
                # update status
                stimulus_txt_2.status = STARTED
                stimulus_txt_2.setAutoDraw(True)
            
            # if stimulus_txt_2 is active this frame...
            if stimulus_txt_2.status == STARTED:
                # update params
                pass
            
            # *phase_txt_read* updates
            
            # if phase_txt_read is starting this frame...
            if phase_txt_read.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                phase_txt_read.frameNStart = frameN  # exact frame index
                phase_txt_read.tStart = t  # local t and not account for scr refresh
                phase_txt_read.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(phase_txt_read, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'phase_txt_read.started')
                # update status
                phase_txt_read.status = STARTED
                phase_txt_read.setAutoDraw(True)
            
            # if phase_txt_read is active this frame...
            if phase_txt_read.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=read,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                read.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if read.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in read.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "read" ---
        for thisComponent in read.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for read
        read.tStop = globalClock.getTime(format='float')
        read.tStopRefresh = tThisFlipGlobal
        thisExp.addData('read.stopped', read.tStop)
        # the Routine "read" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[stimulus_go_txt, phase_txt_trial],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        speech_window = 2.0
        trial_start_time = globalClock.getTime()
        
        current_trial_index += 1
        decoder_result_for_trial = None
        expected_result_json_path = None
        
        stimulus_go_txt.setText(target_word)
        win.callOnFlip(outlet.push_sample, [f"IMAGINED_GO:{target_word}"])
        thisExp.addData('current_trial_index', current_trial_index)
        stimulus_go_txt.setText(target_word)
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        thisExp.currentRoutine = trial
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_online_imagined, 'status') and thisTrials_online_imagined.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code
            if t >= speech_window: 
                continueRoutine = False
            
            # *stimulus_go_txt* updates
            
            # if stimulus_go_txt is starting this frame...
            if stimulus_go_txt.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                stimulus_go_txt.frameNStart = frameN  # exact frame index
                stimulus_go_txt.tStart = t  # local t and not account for scr refresh
                stimulus_go_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stimulus_go_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stimulus_go_txt.started')
                # update status
                stimulus_go_txt.status = STARTED
                stimulus_go_txt.setAutoDraw(True)
            
            # if stimulus_go_txt is active this frame...
            if stimulus_go_txt.status == STARTED:
                # update params
                pass
            
            # *phase_txt_trial* updates
            
            # if phase_txt_trial is starting this frame...
            if phase_txt_trial.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                phase_txt_trial.frameNStart = frameN  # exact frame index
                phase_txt_trial.tStart = t  # local t and not account for scr refresh
                phase_txt_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(phase_txt_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'phase_txt_trial.started')
                # update status
                phase_txt_trial.status = STARTED
                phase_txt_trial.setAutoDraw(True)
            
            # if phase_txt_trial is active this frame...
            if phase_txt_trial.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # Run 'End Routine' code from code
        trial_end_time = globalClock.getTime()
        completed_online_trials += 1
        thisExp.addData('trial_duration', trial_end_time - trial_start_time)
        
        # Do NOT overwrite predicted_word here.
        # The decoder result will be read in decode_wait.
        outlet.push_sample([f"IMAGINED_END:{target_word}"])
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "decode_wait" ---
        # create an object to store info about Routine decode_wait
        decode_wait = data.Routine(
            name='decode_wait',
            components=[decode_text],
        )
        decode_wait.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from decode_code
        decoder_result_for_trial = None
        decode_wait_start_time = globalClock.getTime()
        decode_timeout_sec = 10.0
        decoder_read_error = ''
        
        predicted_word = ''
        prediction_confidence = ''
        decoder_latency_ms = ''
        decoder_status = 'waiting'
        
        if not online_decoder_results_dir and PREPARED_RUN is not None:
            online_decoder_results_dir = PREPARED_RUN.get('online_results_dir')
        
        if online_decoder_results_dir:
            expected_result_json_path = os.path.join(
                online_decoder_results_dir,
                f"trial_{current_trial_index:03d}.json"
            )
        else:
            expected_result_json_path = None
        
        thisExp.addData('expected_result_json_path', expected_result_json_path if expected_result_json_path else '')
        thisExp.addData('target_word', target_word)
        # store start times for decode_wait
        decode_wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        decode_wait.tStart = globalClock.getTime(format='float')
        decode_wait.status = STARTED
        thisExp.addData('decode_wait.started', decode_wait.tStart)
        decode_wait.maxDuration = None
        # keep track of which components have finished
        decode_waitComponents = decode_wait.components
        for thisComponent in decode_wait.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "decode_wait" ---
        thisExp.currentRoutine = decode_wait
        decode_wait.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_online_imagined, 'status') and thisTrials_online_imagined.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from decode_code
            if expected_result_json_path is not None and os.path.exists(expected_result_json_path):
                try:
                    with open(expected_result_json_path, 'r', encoding='utf-8') as f:
                        decoder_result_for_trial = json.load(f)
                    decoder_read_error = ''
                except (PermissionError, OSError, json.JSONDecodeError, ValueError) as exc:
                    # Windows can briefly lock the JSON while the decoder is writing it,
                    # or PsychoPy can catch the file before it is fully flushed. Do not
                    # crash the experiment; keep waiting and retry next frame until timeout.
                    decoder_result_for_trial = None
                    decoder_read_error = f"{type(exc).__name__}: {exc}"
                    decoder_status = 'read_retry'
                    feedback_text = (
                        f"Target: {target_word}\n"
                        f"Waiting for decoder result...\n"
                        f"{decoder_read_error}"
                    )
            
                if decoder_result_for_trial is not None:
                    predicted_word = decoder_result_for_trial.get('predicted_word', 'NO_RESULT')
                    prediction_confidence = decoder_result_for_trial.get('confidence', '')
                    decoder_latency_ms = decoder_result_for_trial.get('latency_ms', '')
                    decoder_status = decoder_result_for_trial.get('status', 'unknown')
            
                    if decoder_status == "ok":
                        feedback_text = (
                            f"Target: {target_word}\n"
                            f"Prediction: {predicted_word}\n"
                            f"Confidence: {prediction_confidence}"
                        )
                    else:
                        feedback_text = (
                            f"Target: {target_word}\n"
                            f"Prediction: {predicted_word}\n"
                            f"Confidence: {prediction_confidence}\n"
                            f"Decoder status: {decoder_status}"
                        )
            
                    continueRoutine = False
            
            elif globalClock.getTime() - decode_wait_start_time >= decode_timeout_sec:
                predicted_word = 'TIMEOUT'
                prediction_confidence = ''
                decoder_latency_ms = ''
                decoder_status = 'timeout'
            
                feedback_text = (
                    f"Target: {target_word}\n"
                    f"Prediction: {predicted_word}\n"
                    f"Decoder status: timeout"
                )
            
                continueRoutine = False
            
            # If the file exists but is temporarily unreadable, the retry branch above keeps
            # continueRoutine=True until the normal timeout condition is reached.
            
            # *decode_text* updates
            
            # if decode_text is starting this frame...
            if decode_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                decode_text.frameNStart = frameN  # exact frame index
                decode_text.tStart = t  # local t and not account for scr refresh
                decode_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(decode_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'decode_text.started')
                # update status
                decode_text.status = STARTED
                decode_text.setAutoDraw(True)
            
            # if decode_text is active this frame...
            if decode_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=decode_wait,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                decode_wait.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if decode_wait.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in decode_wait.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "decode_wait" ---
        for thisComponent in decode_wait.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for decode_wait
        decode_wait.tStop = globalClock.getTime(format='float')
        decode_wait.tStopRefresh = tThisFlipGlobal
        thisExp.addData('decode_wait.stopped', decode_wait.tStop)
        # Run 'End Routine' code from decode_code
        thisExp.addData('predicted_word', predicted_word)
        thisExp.addData('prediction_confidence', prediction_confidence)
        thisExp.addData('decoder_status', decoder_status)
        thisExp.addData('decoder_latency_ms', decoder_latency_ms)
        thisExp.addData('decoder_read_error', decoder_read_error if 'decoder_read_error' in globals() or 'decoder_read_error' in locals() else '')
        # the Routine "decode_wait" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "online_feedback" ---
        # create an object to store info about Routine online_feedback
        online_feedback = data.Routine(
            name='online_feedback',
            components=[feedback_txt],
        )
        online_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback_txt.setText(feedback_text)
        # store start times for online_feedback
        online_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        online_feedback.tStart = globalClock.getTime(format='float')
        online_feedback.status = STARTED
        thisExp.addData('online_feedback.started', online_feedback.tStart)
        online_feedback.maxDuration = None
        # keep track of which components have finished
        online_feedbackComponents = online_feedback.components
        for thisComponent in online_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "online_feedback" ---
        thisExp.currentRoutine = online_feedback
        online_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_online_imagined, 'status') and thisTrials_online_imagined.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_txt* updates
            
            # if feedback_txt is starting this frame...
            if feedback_txt.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                feedback_txt.frameNStart = frameN  # exact frame index
                feedback_txt.tStart = t  # local t and not account for scr refresh
                feedback_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_txt.started')
                # update status
                feedback_txt.status = STARTED
                feedback_txt.setAutoDraw(True)
            
            # if feedback_txt is active this frame...
            if feedback_txt.status == STARTED:
                # update params
                pass
            
            # if feedback_txt is stopping this frame...
            if feedback_txt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_txt.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_txt.tStop = t  # not accounting for scr refresh
                    feedback_txt.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_txt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_txt.stopped')
                    # update status
                    feedback_txt.status = FINISHED
                    feedback_txt.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=online_feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                online_feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if online_feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in online_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "online_feedback" ---
        for thisComponent in online_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for online_feedback
        online_feedback.tStop = globalClock.getTime(format='float')
        online_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('online_feedback.stopped', online_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if online_feedback.maxDurationReached:
            routineTimer.addTime(-online_feedback.maxDuration)
        elif online_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "mini_break" ---
        # create an object to store info about Routine mini_break
        mini_break = data.Routine(
            name='mini_break',
            components=[break_key, break_txt],
        )
        mini_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from break_code
        # Show a rest screen after every 50 completed online imagined trials, except after the final trial.
        # This is safer for long validation runs: with 450 trials, this gives about 8 rests.
        break_every_n_trials = 50
        try:
            total_imagined_trials = int(trials_online_imagined.nTotal)
        except Exception:
            total_imagined_trials = 0
        
        show_mini_break = (
            completed_online_trials > 0
            and completed_online_trials % break_every_n_trials == 0
            and (total_imagined_trials <= 0 or completed_online_trials < total_imagined_trials)
        )
        
        if not show_mini_break:
            marker_txt = None
            continueRoutine = False
        else:
            force_window_focus(win)
            marker_txt = f"IMAGINED_REST_END:{completed_online_trials}"
            win.callOnFlip(outlet.push_sample, [f"IMAGINED_REST_START:{completed_online_trials}"])
            print(f"[REST] {completed_online_trials} imagined trials completed. Press SPACEBAR to continue.", flush=True)
        
        # create starting attributes for break_key
        break_key.keys = []
        break_key.rt = []
        _break_key_allKeys = []
        # store start times for mini_break
        mini_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mini_break.tStart = globalClock.getTime(format='float')
        mini_break.status = STARTED
        thisExp.addData('mini_break.started', mini_break.tStart)
        mini_break.maxDuration = None
        # keep track of which components have finished
        mini_breakComponents = mini_break.components
        for thisComponent in mini_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mini_break" ---
        thisExp.currentRoutine = mini_break
        mini_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_online_imagined, 'status') and thisTrials_online_imagined.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_key* updates
            waitOnFlip = False
            
            # if break_key is starting this frame...
            if break_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_key.frameNStart = frameN  # exact frame index
                break_key.tStart = t  # local t and not account for scr refresh
                break_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_key.started')
                # update status
                break_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(break_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(break_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if break_key.status == STARTED and not waitOnFlip:
                theseKeys = break_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _break_key_allKeys.extend(theseKeys)
                if len(_break_key_allKeys):
                    break_key.keys = _break_key_allKeys[-1].name  # just the last key pressed
                    break_key.rt = _break_key_allKeys[-1].rt
                    break_key.duration = _break_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *break_txt* updates
            
            # if break_txt is starting this frame...
            if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_txt.frameNStart = frameN  # exact frame index
                break_txt.tStart = t  # local t and not account for scr refresh
                break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_txt.started')
                # update status
                break_txt.status = STARTED
                break_txt.setAutoDraw(True)
            
            # if break_txt is active this frame...
            if break_txt.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=mini_break,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                mini_break.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if mini_break.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in mini_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mini_break" ---
        for thisComponent in mini_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mini_break
        mini_break.tStop = globalClock.getTime(format='float')
        mini_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mini_break.stopped', mini_break.tStop)
        # Run 'End Routine' code from break_code
        if marker_txt:
            win.callOnFlip(outlet.push_sample, [marker_txt])
        
        # check responses
        if break_key.keys in ['', [], None]:  # No response was made
            break_key.keys = None
        trials_online_imagined.addData('break_key.keys',break_key.keys)
        if break_key.keys != None:  # we had a response
            trials_online_imagined.addData('break_key.rt', break_key.rt)
            trials_online_imagined.addData('break_key.duration', break_key.duration)
        # the Routine "mini_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_online_imagined as finished
        if hasattr(thisTrials_online_imagined, 'status'):
            thisTrials_online_imagined.status = FINISHED
        # if awaiting a pause, pause now
        if trials_online_imagined.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_online_imagined.status = STARTED
        thisExp.nextEntry()
        
    # completed 45.0 repeats of 'trials_online_imagined'
    trials_online_imagined.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Stop_record" ---
    # create an object to store info about Routine Stop_record
    Stop_record = data.Routine(
        name='Stop_record',
        components=[],
    )
    Stop_record.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from stop_code
    win.callOnFlip(outlet.push_sample, [f"{block}_BLOCK_END"])
    print(f"{block} block finished")
    # store start times for Stop_record
    Stop_record.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Stop_record.tStart = globalClock.getTime(format='float')
    Stop_record.status = STARTED
    thisExp.addData('Stop_record.started', Stop_record.tStart)
    Stop_record.maxDuration = None
    # keep track of which components have finished
    Stop_recordComponents = Stop_record.components
    for thisComponent in Stop_record.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Stop_record" ---
    thisExp.currentRoutine = Stop_record
    Stop_record.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Stop_record,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Stop_record.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Stop_record.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Stop_record.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Stop_record" ---
    for thisComponent in Stop_record.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Stop_record
    Stop_record.tStop = globalClock.getTime(format='float')
    Stop_record.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Stop_record.stopped', Stop_record.tStop)
    # Run 'End Routine' code from stop_code
    try:
        stop_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stop_socket.settimeout(2.0)
        stop_socket.connect(("localhost", 22345))
        stop_socket.sendall(b"stop\n")
        stop_socket.close()
        print("Loop finished. LabRecorder stopped and block file saved.")
    except:
        pass
    thisExp.nextEntry()
    # the Routine "Stop_record" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Finish" ---
    # create an object to store info about Routine Finish
    Finish = data.Routine(
        name='Finish',
        components=[text_2, key_resp],
    )
    Finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_2.setText(run_summary_text)
    # Run 'Begin Routine' code from finish_code
    summary_path = None
    summary_payload = None
    
    if (not online_decoder_results_dir) and (PREPARED_RUN is not None):
        online_decoder_results_dir = PREPARED_RUN.get('online_results_dir')
    
    if online_decoder_results_dir:
        summary_path = os.path.join(online_decoder_results_dir, 'run_summary.json')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_payload = json.load(f)
            except Exception:
                summary_payload = None
    
    best_cosine_text = 'N/A'
    if brainprint_best_match_cosine is not None:
        best_cosine_text = f"{brainprint_best_match_cosine:.4f}"
    
    best_match_name = 'N/A'
    if brainprint_best_match_file:
        best_match_name = os.path.basename(brainprint_best_match_file)
    
    if summary_payload is not None:
        n_trials = summary_payload.get('n_trials', completed_online_trials)
        n_correct = summary_payload.get('n_correct', 'N/A')
        accuracy = summary_payload.get('accuracy', None)
        mean_conf = summary_payload.get('mean_confidence', None)
        mean_lat = summary_payload.get('mean_latency_ms', None)
    
        acc_text = f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "N/A"
        conf_text = f"{mean_conf:.3f}" if isinstance(mean_conf, (int, float)) else "N/A"
        lat_text = f"{mean_lat:.1f} ms" if isinstance(mean_lat, (int, float)) else "N/A"
    
        run_summary_text = (
            "Online imagined run complete.\n\n"
            f"Trials completed: {n_trials}\n"
            f"Correct: {n_correct}\n"
            f"Accuracy: {acc_text}\n"
            f"Mean confidence: {conf_text}\n"
            f"Mean latency: {lat_text}\n\n"
            f"Brainprint status: {brainprint_status}\n"
            f"Adaptation allowed: {adaptation_allowed}\n"
            f"Adaptation reason: {adaptation_reason}\n"
            f"Best match: {best_match_name}\n"
            f"Best cosine: {best_cosine_text}\n\n"
            "Press SPACE to close."
        )
    else:
        run_summary_text = (
            "Online imagined run complete.\n\n"
            f"Trials completed: {completed_online_trials}\n\n"
            f"Brainprint status: {brainprint_status}\n"
            f"Adaptation allowed: {adaptation_allowed}\n"
            f"Adaptation reason: {adaptation_reason}\n"
            f"Best match: {best_match_name}\n"
            f"Best cosine: {best_cosine_text}\n\n"
            "Decoder summary not found.\n\n"
            "Press SPACE to close."
        )
    
    text_2.setText(run_summary_text)
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Finish
    Finish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Finish.tStart = globalClock.getTime(format='float')
    Finish.status = STARTED
    thisExp.addData('Finish.started', Finish.tStart)
    Finish.maxDuration = None
    # keep track of which components have finished
    FinishComponents = Finish.components
    for thisComponent in Finish.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Finish" ---
    thisExp.currentRoutine = Finish
    Finish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space', 'escape'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Finish,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Finish.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Finish.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Finish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Finish" ---
    for thisComponent in Finish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Finish
    Finish.tStop = globalClock.getTime(format='float')
    Finish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Finish.stopped', Finish.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Finish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from decode_code
    try: 
        if online_decoder_proc is not None and online_decoder_proc.poll() is None: 
            online_decoder_proc.terminate() 
            try: 
                online_decoder_proc.wait(timeout=2.0) 
            except Exception: 
                online_decoder_proc.kill() 
            print("Online decoder service terminated cleanly.") 
    except Exception as e: 
        print(f"Could not terminate online decoder service cleanly: {e}")
    # Run 'End Experiment' code from feedback_code
    # Destroy the LSL stream to release the background thread
    try:
        del outlet
        del info
        print("LSL marker stream closed cleanly.")
    except:
        pass
    # Run 'End Experiment' code from finish_code
    # Destroy the LSL stream to release the background thread
    try:
        del outlet
        del info
        print("LSL marker stream closed cleanly.")
    except:
        pass
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

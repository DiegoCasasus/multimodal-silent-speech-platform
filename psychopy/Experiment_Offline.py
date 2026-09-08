#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.1),
    on May 19, 2026, at 13:07
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BRIDGE_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "latest_prepared_run.json"),
]

LABRECORDER_BASE_ROOT = os.path.join(PROJECT_ROOT, "recordings")

LABRECORDER_EXE_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "LabRecorder.exe"),
    os.path.join(SCRIPT_DIR, "LabRecorder", "LabRecorder.exe"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "LabRecorder.exe"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "LabRecorder", "LabRecorder.exe"),
    os.path.join(PROJECT_ROOT, "LabRecorder.exe"),
    os.path.join(PROJECT_ROOT, "LabRecorder", "LabRecorder.exe"),
    r"C:\Program Files\LabRecorder\LabRecorder.exe",
    r"C:\Program Files (x86)\LabRecorder\LabRecorder.exe",
]

labrecorder_proc = None
PREPARED_RUN = None
PREPARED_RUN_SOURCE = None


def force_window_focus(win):
    try:
        win.winHandle.activate()
        core.wait(0.15)
    except Exception as e:
        print(f"Could not force window focus: {e}")


def load_prepared_run_bridge():
    global PREPARED_RUN_SOURCE

    for candidate in BRIDGE_CANDIDATES:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                PREPARED_RUN_SOURCE = candidate
                print(f"Loaded GUI bridge file: {candidate}")
                return payload
            except Exception as e:
                print(f"Could not read GUI bridge file at {candidate}: {e}")

    print("No GUI bridge file found.")
    return None


def apply_bridge_to_expinfo(expInfo, prepared_run):
    if not prepared_run:
        return expInfo

    expInfo["participant"] = prepared_run.get("participant_id", expInfo.get("participant", "001"))
    expInfo["session"] = prepared_run.get("session_id", expInfo.get("session", "001"))
    expInfo["run"] = prepared_run.get("run_id", expInfo.get("run", "001"))
    return expInfo


def resolve_block_name(block_num):
    if block_num == 1:
        return "calibration"
    elif block_num == 2:
        return "silent"
    elif block_num == 3:
        return "imagined"
    return f"block{block_num}"


def build_block_xdf_template(expInfo, block):
    participant = str(expInfo.get("participant", "test_subject")).replace(" ", "_")
    session = str(expInfo.get("session", "001")).replace(" ", "_")
    return f"exp_sub_{participant}_ses_{session}_bl_{block}.xdf"


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
        recording_block_paths = PREPARED_RUN.get("recording_block_paths", {})

        block_path = recording_block_paths.get(block)
        if block_path:
            block_path = os.path.normpath(block_path)
            file_root = os.path.dirname(block_path)
            template = os.path.basename(block_path)

        elif PREPARED_RUN.get("expected_xdf_path"):
            expected_xdf_path = os.path.normpath(PREPARED_RUN["expected_xdf_path"])
            file_root = os.path.dirname(expected_xdf_path)
            template = build_block_xdf_template(expInfo, block)

    if file_root is None:
        file_root = os.path.join(LABRECORDER_BASE_ROOT, block)
        template = build_block_xdf_template(expInfo, block)

    os.makedirs(file_root, exist_ok=True)
    file_root = os.path.normpath(file_root) + "\\"

    cmd = f"filename {{root:{file_root}}} {{template:{template}}}"
    print("Sending LabRecorder filename command:", cmd)
    return cmd


def labrecorder_is_listening(host="localhost", port=22345, timeout=0.5):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def find_labrecorder_exe():
    from_path = shutil.which("LabRecorder.exe") or shutil.which("LabRecorder")
    if from_path:
        return from_path

    for exe in LABRECORDER_EXE_CANDIDATES:
        if exe and os.path.exists(exe):
            return exe

    search_roots = [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR), PROJECT_ROOT]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth > 3:
                dirnames[:] = []
                continue

            if "LabRecorder.exe" in filenames:
                return os.path.join(dirpath, "LabRecorder.exe")

    return None


def ensure_labrecorder_running():
    global labrecorder_proc

    if labrecorder_is_listening():
        print("LabRecorder already running.")
        return True

    exe = find_labrecorder_exe()
    if exe is None:
        print("Could not find LabRecorder.exe.")
        print("Checked explicit candidates:", LABRECORDER_EXE_CANDIDATES)
        return False

    try:
        print(f"Launching LabRecorder from: {exe}")
        labrecorder_proc = subprocess.Popen([exe])

        for _ in range(20):
            time.sleep(0.5)
            if labrecorder_is_listening():
                print("LabRecorder launched successfully and remote control is available.")
                return True

        print("LabRecorder launched, but remote-control port 22345 did not become available.")
    except Exception as e:
        print(f"Failed to launch LabRecorder from {exe}: {e}")

    return False



# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.1'
expName = 'exp'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
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
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
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
            size=_winSize, fullscr=_fullScr, screen=0,
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
    global PREPARED_RUN, PREPARED_RUN_SOURCE
    PREPARED_RUN = load_prepared_run_bridge()
    expInfo = apply_bridge_to_expinfo(expInfo, PREPARED_RUN)
    
    print("PREPARED_RUN =", PREPARED_RUN)
    print("expInfo after bridge =", expInfo)
    
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
    
    # 3. Create the marker stream ONLY ONCE
    info = StreamInfo("PsychoPy_Markers", "Markers", 1, 0, "string", "my_silent_speech_exp")
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
    
    # --- Initialize components for Routine "Instructions_silent" ---
    instructions_key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    inst_silent_txt = visual.TextStim(win=win, name='inst_silent_txt',
        text='Look at the center cross.\n\nA white word will appear.\n\nWhen the word turns green, perform the silent speech (articulate the word without sound).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Start_record" ---
    
    # --- Initialize components for Routine "wait" ---
    ready_txt = visual.TextStim(win=win, name='ready_txt',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "read" ---
    stimulus_txt_2 = visual.TextStim(win=win, name='stimulus_txt_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    stimulus_go_txt = visual.TextStim(win=win, name='stimulus_go_txt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color=(0.1294, 0.8667, 0.1294), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "mini_break" ---
    break_key = keyboard.Keyboard(deviceName='defaultKeyboard')
    break_txt = visual.TextStim(win=win, name='break_txt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Stop_record" ---
    
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
    ready_txt = visual.TextStim(win=win, name='ready_txt',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "read" ---
    stimulus_txt_2 = visual.TextStim(win=win, name='stimulus_txt_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    stimulus_go_txt = visual.TextStim(win=win, name='stimulus_go_txt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color=(0.1294, 0.8667, 0.1294), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "mini_break" ---
    break_key = keyboard.Keyboard(deviceName='defaultKeyboard')
    break_txt = visual.TextStim(win=win, name='break_txt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Stop_record" ---
    
    # --- Initialize components for Routine "Finish" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Experiment finished',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
            #print("Sending LabRecorder filename command:", file_cmd)
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
    
    # --- Prepare to start Routine "Instructions_silent" ---
    # create an object to store info about Routine Instructions_silent
    Instructions_silent = data.Routine(
        name='Instructions_silent',
        components=[instructions_key_resp, inst_silent_txt],
    )
    Instructions_silent.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instructions_key_resp
    instructions_key_resp.keys = []
    instructions_key_resp.rt = []
    _instructions_key_resp_allKeys = []
    # Run 'Begin Routine' code from code_silent_instructions
    force_window_focus(win)
    # store start times for Instructions_silent
    Instructions_silent.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions_silent.tStart = globalClock.getTime(format='float')
    Instructions_silent.status = STARTED
    thisExp.addData('Instructions_silent.started', Instructions_silent.tStart)
    Instructions_silent.maxDuration = None
    # keep track of which components have finished
    Instructions_silentComponents = Instructions_silent.components
    for thisComponent in Instructions_silent.components:
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
    
    # --- Run Routine "Instructions_silent" ---
    thisExp.currentRoutine = Instructions_silent
    Instructions_silent.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_key_resp* updates
        waitOnFlip = False
        
        # if instructions_key_resp is starting this frame...
        if instructions_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_key_resp.frameNStart = frameN  # exact frame index
            instructions_key_resp.tStart = t  # local t and not account for scr refresh
            instructions_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_key_resp.started')
            # update status
            instructions_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instructions_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instructions_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instructions_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instructions_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instructions_key_resp_allKeys.extend(theseKeys)
            if len(_instructions_key_resp_allKeys):
                instructions_key_resp.keys = _instructions_key_resp_allKeys[-1].name  # just the last key pressed
                instructions_key_resp.rt = _instructions_key_resp_allKeys[-1].rt
                instructions_key_resp.duration = _instructions_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *inst_silent_txt* updates
        
        # if inst_silent_txt is starting this frame...
        if inst_silent_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            inst_silent_txt.frameNStart = frameN  # exact frame index
            inst_silent_txt.tStart = t  # local t and not account for scr refresh
            inst_silent_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(inst_silent_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'inst_silent_txt.started')
            # update status
            inst_silent_txt.status = STARTED
            inst_silent_txt.setAutoDraw(True)
        
        # if inst_silent_txt is active this frame...
        if inst_silent_txt.status == STARTED:
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
                currentRoutine=Instructions_silent,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instructions_silent.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instructions_silent.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instructions_silent.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions_silent" ---
    for thisComponent in Instructions_silent.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions_silent
    Instructions_silent.tStop = globalClock.getTime(format='float')
    Instructions_silent.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions_silent.stopped', Instructions_silent.tStop)
    # check responses
    if instructions_key_resp.keys in ['', [], None]:  # No response was made
        instructions_key_resp.keys = None
    thisExp.addData('instructions_key_resp.keys',instructions_key_resp.keys)
    if instructions_key_resp.keys != None:  # we had a response
        thisExp.addData('instructions_key_resp.rt', instructions_key_resp.rt)
        thisExp.addData('instructions_key_resp.duration', instructions_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Instructions_silent" was not non-slip safe, so reset the non-slip timer
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
            #print("Sending LabRecorder filename command:", file_cmd)
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
    break_loop = data.TrialHandler2(
        name='break_loop',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(break_loop)  # add the loop to the experiment
    thisBreak_loop = break_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBreak_loop.rgb)
    if thisBreak_loop != None:
        for paramName in thisBreak_loop:
            globals()[paramName] = thisBreak_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBreak_loop in break_loop:
        break_loop.status = STARTED
        if hasattr(thisBreak_loop, 'status'):
            thisBreak_loop.status = STARTED
        currentLoop = break_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBreak_loop.rgb)
        if thisBreak_loop != None:
            for paramName in thisBreak_loop:
                globals()[paramName] = thisBreak_loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_silent = data.TrialHandler2(
            name='trials_silent',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Conditions.xlsx'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(trials_silent)  # add the loop to the experiment
        thisTrials_silent = trials_silent.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_silent.rgb)
        if thisTrials_silent != None:
            for paramName in thisTrials_silent:
                globals()[paramName] = thisTrials_silent[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_silent in trials_silent:
            trials_silent.status = STARTED
            if hasattr(thisTrials_silent, 'status'):
                thisTrials_silent.status = STARTED
            currentLoop = trials_silent
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_silent.rgb)
            if thisTrials_silent != None:
                for paramName in thisTrials_silent:
                    globals()[paramName] = thisTrials_silent[paramName]
            
            # --- Prepare to start Routine "wait" ---
            # create an object to store info about Routine wait
            wait = data.Routine(
                name='wait',
                components=[ready_txt],
            )
            wait.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_2
            # Ejemplo: Duración aleatoria entre 1 y 3 segundos
            import numpy as np
            duracion_estimulo = np.random.uniform(0.5, 1.5)
            
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
                if hasattr(thisTrials_silent, 'status') and thisTrials_silent.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *ready_txt* updates
                
                # if ready_txt is starting this frame...
                if ready_txt.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    ready_txt.frameNStart = frameN  # exact frame index
                    ready_txt.tStart = t  # local t and not account for scr refresh
                    ready_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ready_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ready_txt.started')
                    # update status
                    ready_txt.status = STARTED
                    ready_txt.setAutoDraw(True)
                
                # if ready_txt is active this frame...
                if ready_txt.status == STARTED:
                    # update params
                    pass
                
                # if ready_txt is stopping this frame...
                if ready_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ready_txt.tStartRefresh + duracion_estimulo-frameTolerance:
                        # keep track of stop time/frame for later
                        ready_txt.tStop = t  # not accounting for scr refresh
                        ready_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        ready_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ready_txt.stopped')
                        # update status
                        ready_txt.status = FINISHED
                        ready_txt.setAutoDraw(False)
                
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
                components=[stimulus_txt_2],
            )
            read.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
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
            while continueRoutine and routineTimer.getTime() < 1.5:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_silent, 'status') and thisTrials_silent.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                
                # if stimulus_txt_2 is stopping this frame...
                if stimulus_txt_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stimulus_txt_2.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        stimulus_txt_2.tStop = t  # not accounting for scr refresh
                        stimulus_txt_2.tStopRefresh = tThisFlipGlobal  # on global time
                        stimulus_txt_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stimulus_txt_2.stopped')
                        # update status
                        stimulus_txt_2.status = FINISHED
                        stimulus_txt_2.setAutoDraw(False)
                
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if read.maxDurationReached:
                routineTimer.addTime(-read.maxDuration)
            elif read.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[stimulus_go_txt],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            # NOTE: Replace 'target_word' with the actual name of your column in your Excel/CSV conditions file!
            marker_text = f"Cue_Start: {target_word}"
            
            # Send the marker exactly when the screen refreshes to show the word
            win.callOnFlip(outlet.push_sample, [marker_text])
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
            while continueRoutine and routineTimer.getTime() < 2.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_silent, 'status') and thisTrials_silent.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                
                # if stimulus_go_txt is stopping this frame...
                if stimulus_go_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stimulus_go_txt.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        stimulus_go_txt.tStop = t  # not accounting for scr refresh
                        stimulus_go_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        stimulus_go_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stimulus_go_txt.stopped')
                        # update status
                        stimulus_go_txt.status = FINISHED
                        stimulus_go_txt.setAutoDraw(False)
                
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            # mark thisTrials_silent as finished
            if hasattr(thisTrials_silent, 'status'):
                thisTrials_silent.status = FINISHED
            # if awaiting a pause, pause now
            if trials_silent.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_silent.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_silent'
        trials_silent.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
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
        force_window_focus(win)
        
        if block == "silent":
            current_batch = silent_batch
            silent_batch += 1
            marker_txt = "SILENT_LOOP_BREAK_END"
        elif block == "imagined":
            current_batch = imagined_batch
            imagined_batch += 1
            marker_txt = "IMAGINED_LOOP_BREAK_END"
        else:
            current_batch = 0
            marker_txt = f"{str(block).upper()}_LOOP_BREAK_END"
        
        break_text = f"Loop {current_batch} finished!\n\nTake a short rest, blink, and swallow.\nPress SPACEBAR to start the next 10 words."
        
        # create starting attributes for break_key
        break_key.keys = []
        break_key.rt = []
        _break_key_allKeys = []
        break_txt.setText(break_text)
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
            if hasattr(thisBreak_loop, 'status') and thisBreak_loop.status == STOPPING:
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
        win.callOnFlip(outlet.push_sample, [marker_txt])
        # check responses
        if break_key.keys in ['', [], None]:  # No response was made
            break_key.keys = None
        break_loop.addData('break_key.keys',break_key.keys)
        if break_key.keys != None:  # we had a response
            break_loop.addData('break_key.rt', break_key.rt)
            break_loop.addData('break_key.duration', break_key.duration)
        # the Routine "mini_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBreak_loop as finished
        if hasattr(thisBreak_loop, 'status'):
            thisBreak_loop.status = FINISHED
        # if awaiting a pause, pause now
        if break_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            break_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'break_loop'
    break_loop.status = FINISHED
    
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
            #print("Sending LabRecorder filename command:", file_cmd)
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
    break_loop2 = data.TrialHandler2(
        name='break_loop2',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(break_loop2)  # add the loop to the experiment
    thisBreak_loop2 = break_loop2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBreak_loop2.rgb)
    if thisBreak_loop2 != None:
        for paramName in thisBreak_loop2:
            globals()[paramName] = thisBreak_loop2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBreak_loop2 in break_loop2:
        break_loop2.status = STARTED
        if hasattr(thisBreak_loop2, 'status'):
            thisBreak_loop2.status = STARTED
        currentLoop = break_loop2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBreak_loop2.rgb)
        if thisBreak_loop2 != None:
            for paramName in thisBreak_loop2:
                globals()[paramName] = thisBreak_loop2[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_imagined = data.TrialHandler2(
            name='trials_imagined',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Conditions.xlsx'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(trials_imagined)  # add the loop to the experiment
        thisTrials_imagined = trials_imagined.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_imagined.rgb)
        if thisTrials_imagined != None:
            for paramName in thisTrials_imagined:
                globals()[paramName] = thisTrials_imagined[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_imagined in trials_imagined:
            trials_imagined.status = STARTED
            if hasattr(thisTrials_imagined, 'status'):
                thisTrials_imagined.status = STARTED
            currentLoop = trials_imagined
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_imagined.rgb)
            if thisTrials_imagined != None:
                for paramName in thisTrials_imagined:
                    globals()[paramName] = thisTrials_imagined[paramName]
            
            # --- Prepare to start Routine "wait" ---
            # create an object to store info about Routine wait
            wait = data.Routine(
                name='wait',
                components=[ready_txt],
            )
            wait.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_2
            # Ejemplo: Duración aleatoria entre 1 y 3 segundos
            import numpy as np
            duracion_estimulo = np.random.uniform(0.5, 1.5)
            
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
                if hasattr(thisTrials_imagined, 'status') and thisTrials_imagined.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *ready_txt* updates
                
                # if ready_txt is starting this frame...
                if ready_txt.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    ready_txt.frameNStart = frameN  # exact frame index
                    ready_txt.tStart = t  # local t and not account for scr refresh
                    ready_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ready_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ready_txt.started')
                    # update status
                    ready_txt.status = STARTED
                    ready_txt.setAutoDraw(True)
                
                # if ready_txt is active this frame...
                if ready_txt.status == STARTED:
                    # update params
                    pass
                
                # if ready_txt is stopping this frame...
                if ready_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ready_txt.tStartRefresh + duracion_estimulo-frameTolerance:
                        # keep track of stop time/frame for later
                        ready_txt.tStop = t  # not accounting for scr refresh
                        ready_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        ready_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ready_txt.stopped')
                        # update status
                        ready_txt.status = FINISHED
                        ready_txt.setAutoDraw(False)
                
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
                components=[stimulus_txt_2],
            )
            read.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
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
            while continueRoutine and routineTimer.getTime() < 1.5:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_imagined, 'status') and thisTrials_imagined.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                
                # if stimulus_txt_2 is stopping this frame...
                if stimulus_txt_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stimulus_txt_2.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        stimulus_txt_2.tStop = t  # not accounting for scr refresh
                        stimulus_txt_2.tStopRefresh = tThisFlipGlobal  # on global time
                        stimulus_txt_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stimulus_txt_2.stopped')
                        # update status
                        stimulus_txt_2.status = FINISHED
                        stimulus_txt_2.setAutoDraw(False)
                
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if read.maxDurationReached:
                routineTimer.addTime(-read.maxDuration)
            elif read.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[stimulus_go_txt],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            # NOTE: Replace 'target_word' with the actual name of your column in your Excel/CSV conditions file!
            marker_text = f"Cue_Start: {target_word}"
            
            # Send the marker exactly when the screen refreshes to show the word
            win.callOnFlip(outlet.push_sample, [marker_text])
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
            while continueRoutine and routineTimer.getTime() < 2.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_imagined, 'status') and thisTrials_imagined.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                
                # if stimulus_go_txt is stopping this frame...
                if stimulus_go_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stimulus_go_txt.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        stimulus_go_txt.tStop = t  # not accounting for scr refresh
                        stimulus_go_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        stimulus_go_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stimulus_go_txt.stopped')
                        # update status
                        stimulus_go_txt.status = FINISHED
                        stimulus_go_txt.setAutoDraw(False)
                
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            # mark thisTrials_imagined as finished
            if hasattr(thisTrials_imagined, 'status'):
                thisTrials_imagined.status = FINISHED
            # if awaiting a pause, pause now
            if trials_imagined.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_imagined.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_imagined'
        trials_imagined.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
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
        force_window_focus(win)
        
        if block == "silent":
            current_batch = silent_batch
            silent_batch += 1
            marker_txt = "SILENT_LOOP_BREAK_END"
        elif block == "imagined":
            current_batch = imagined_batch
            imagined_batch += 1
            marker_txt = "IMAGINED_LOOP_BREAK_END"
        else:
            current_batch = 0
            marker_txt = f"{str(block).upper()}_LOOP_BREAK_END"
        
        break_text = f"Loop {current_batch} finished!\n\nTake a short rest, blink, and swallow.\nPress SPACEBAR to start the next 10 words."
        
        # create starting attributes for break_key
        break_key.keys = []
        break_key.rt = []
        _break_key_allKeys = []
        break_txt.setText(break_text)
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
            if hasattr(thisBreak_loop2, 'status') and thisBreak_loop2.status == STOPPING:
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
        win.callOnFlip(outlet.push_sample, [marker_txt])
        # check responses
        if break_key.keys in ['', [], None]:  # No response was made
            break_key.keys = None
        break_loop2.addData('break_key.keys',break_key.keys)
        if break_key.keys != None:  # we had a response
            break_loop2.addData('break_key.rt', break_key.rt)
            break_loop2.addData('break_key.duration', break_key.duration)
        # the Routine "mini_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBreak_loop2 as finished
        if hasattr(thisBreak_loop2, 'status'):
            thisBreak_loop2.status = FINISHED
        # if awaiting a pause, pause now
        if break_loop2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            break_loop2.status = STARTED
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'break_loop2'
    break_loop2.status = FINISHED
    
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
        components=[text_2],
    )
    Finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
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
    while continueRoutine and routineTimer.getTime() < 5.5:
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
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
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
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Finish.maxDurationReached:
        routineTimer.addTime(-Finish.maxDuration)
    elif Finish.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.500000)
    thisExp.nextEntry()
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
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
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

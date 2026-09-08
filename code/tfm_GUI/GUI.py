from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from controller import ExperimentController
from models import (
    get_model_display_name,
    get_model_key_from_display_name,
    get_model_spec,
)
from state import AppStatus, ExperimentMode, Modality


class TFMGuiApp:
    """Modern desktop front-end for the TFM multimodal experiment platform.

    This class deliberately keeps the controller/backend contract unchanged.
    The renovation is visual and presentational only.
    """

    COLORS = {
        "bg": "#F3F6FB",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFD",
        "primary": "#2457D6",
        "primary_dark": "#173FA8",
        "primary_soft": "#EAF0FF",
        "text": "#172033",
        "muted": "#667085",
        "border": "#D9E1EC",
        "success": "#228B5E",
        "success_soft": "#E7F6EF",
        "warning": "#B7791F",
        "warning_soft": "#FFF5DA",
        "error": "#C43D4B",
        "error_soft": "#FDECEF",
        "info": "#2D6CDF",
        "info_soft": "#EAF2FF",
    }

    STATUS_STYLE = {
        AppStatus.IDLE: ("IDLE", "muted", "surface_alt"),
        AppStatus.DISCOVERING: ("DISCOVERING", "info", "info_soft"),
        AppStatus.READY: ("READY", "success", "success_soft"),
        AppStatus.RECORDING: ("RECORDING", "warning", "warning_soft"),
        AppStatus.DECODING: ("DECODING", "primary", "primary_soft"),
        AppStatus.ERROR: ("ERROR", "error", "error_soft"),
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TFM Neurotechnology Platform")
        self.root.geometry("1280x780")
        self.root.minsize(1040, 640)
        self.root.configure(bg=self.COLORS["bg"])

        self.controller = ExperimentController()

        self.participant_var = tk.StringVar()
        self.session_var = tk.StringVar()
        self.run_var = tk.StringVar()
        self.experiment_mode_var = tk.StringVar(
            value=self.controller.get_state().selected_experiment_mode
        )
        self.operator_var = tk.StringVar()
        self.site_var = tk.StringVar()

        self.selected_mode_var = tk.StringVar(
            value=self.controller.get_state().selected_mode or "EEG"
        )
        self.selected_model_var = tk.StringVar(
            value=get_model_display_name(self.controller.get_state().selected_model)
        )

        self._last_live_result_signature = None
        self._configure_styles()
        self._build_layout()
        self._render_state()
        self._poll_background_tasks()
        self._poll_live_decoder_result()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.COLORS["bg"])
        style.configure("Surface.TFrame", background=self.COLORS["surface"])
        style.configure("Soft.TFrame", background=self.COLORS["surface_alt"])

        style.configure(
            "Card.TLabelframe",
            background=self.COLORS["surface"],
            bordercolor=self.COLORS["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self.COLORS["surface"],
            foreground=self.COLORS["text"],
            font=("Segoe UI Semibold", 11),
        )

        style.configure(
            "App.TLabel",
            background=self.COLORS["surface"],
            foreground=self.COLORS["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=self.COLORS["surface"],
            foreground=self.COLORS["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=self.COLORS["surface"],
            foreground=self.COLORS["text"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Metric.TLabel",
            background=self.COLORS["surface_alt"],
            foreground=self.COLORS["text"],
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "MetricCaption.TLabel",
            background=self.COLORS["surface_alt"],
            foreground=self.COLORS["muted"],
            font=("Segoe UI", 8),
        )

        style.configure(
            "TEntry",
            fieldbackground=self.COLORS["surface_alt"],
            foreground=self.COLORS["text"],
            bordercolor=self.COLORS["border"],
            lightcolor=self.COLORS["border"],
            darkcolor=self.COLORS["border"],
            padding=6,
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.COLORS["surface_alt"],
            foreground=self.COLORS["text"],
            bordercolor=self.COLORS["border"],
            arrowcolor=self.COLORS["primary"],
            padding=5,
        )

        style.configure(
            "Primary.TButton",
            background=self.COLORS["primary"],
            foreground="#FFFFFF",
            bordercolor=self.COLORS["primary"],
            focusthickness=0,
            padding=(16, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.COLORS["primary_dark"]), ("disabled", "#A7B7E7")],
            foreground=[("disabled", "#EFF3FF")],
        )

        style.configure(
            "Secondary.TButton",
            background=self.COLORS["surface_alt"],
            foreground=self.COLORS["primary"],
            bordercolor=self.COLORS["border"],
            focusthickness=0,
            padding=(14, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", self.COLORS["primary_soft"])],
            bordercolor=[("active", self.COLORS["primary"])],
        )

        style.configure(
            "Danger.TButton",
            background=self.COLORS["error_soft"],
            foreground=self.COLORS["error"],
            bordercolor=self.COLORS["error_soft"],
            padding=(14, 8),
            font=("Segoe UI Semibold", 10),
        )

        style.configure(
            "Horizontal.TSeparator",
            background=self.COLORS["border"],
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _make_card(self, parent, title: str, padding: int = 14) -> ttk.LabelFrame:
        return ttk.LabelFrame(parent, text=title, style="Card.TLabelframe", padding=padding)

    def _make_text_panel(self, parent, height: int = 8) -> tk.Text:
        widget = tk.Text(
            parent,
            wrap="word",
            height=height,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            highlightcolor=self.COLORS["primary"],
            background=self.COLORS["surface_alt"],
            foreground=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            selectbackground=self.COLORS["primary_soft"],
            font=("Consolas", 9),
            padx=10,
            pady=8,
        )
        widget.tag_configure("heading", foreground=self.COLORS["text"], font=("Segoe UI Semibold", 10))
        widget.tag_configure("muted", foreground=self.COLORS["muted"], font=("Segoe UI", 9))
        widget.tag_configure("success", foreground=self.COLORS["success"], font=("Segoe UI Semibold", 9))
        widget.tag_configure("warning", foreground=self.COLORS["warning"], font=("Segoe UI Semibold", 9))
        widget.tag_configure("error", foreground=self.COLORS["error"], font=("Segoe UI Semibold", 9))
        widget.tag_configure("info", foreground=self.COLORS["info"], font=("Segoe UI Semibold", 9))
        return widget

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(22, 18, 22, 18))
        outer.pack(fill="both", expand=True)

        # Header
        header = tk.Frame(outer, bg=self.COLORS["primary"], height=66)
        header.pack(fill="x", pady=(0, 14))
        header.pack_propagate(False)

        title_area = tk.Frame(header, bg=self.COLORS["primary"])
        title_area.pack(side="left", fill="both", expand=True, padx=22, pady=11)
        tk.Label(
            title_area,
            text="Multimodal Neurotechnology Platform",
            bg=self.COLORS["primary"],
            fg="#FFFFFF",
            font=("Segoe UI Semibold", 17),
        ).pack(anchor="w")
        tk.Label(
            title_area,
            text="",
            bg=self.COLORS["primary"],
            fg="#DCE6FF",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        status_area = tk.Frame(header, bg=self.COLORS["primary"])
        status_area.pack(side="right", padx=22)
        self.status_badge = tk.Label(
            status_area,
            text="IDLE",
            bg="#FFFFFF",
            fg=self.COLORS["primary"],
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=7,
        )
        self.status_badge.pack(anchor="e")
        self.error_label = tk.Label(
            status_area,
            text="",
            bg=self.COLORS["primary"],
            fg="#FFD8DE",
            font=("Segoe UI", 9),
            wraplength=340,
            justify="right",
        )
        self.error_label.pack(anchor="e", pady=(5, 0))

        # Two-column workspace
        workspace = ttk.Frame(outer, style="App.TFrame")
        workspace.pack(fill="both", expand=True)
        workspace.columnconfigure(0, weight=5)
        workspace.columnconfigure(1, weight=7)
        workspace.rowconfigure(0, weight=1)

        # The left workflow is scrollable so every control remains reachable on
        # 720p lab displays and smaller laptop screens.
        left_shell = ttk.Frame(workspace, style="App.TFrame")
        left_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        left_shell.rowconfigure(0, weight=1)
        left_shell.columnconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(
            left_shell,
            bg=self.COLORS["bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar = ttk.Scrollbar(
            left_shell, orient="vertical", command=self.left_canvas.yview
        )
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left = ttk.Frame(self.left_canvas, style="App.TFrame")
        self._left_canvas_window = self.left_canvas.create_window(
            (0, 0), window=left, anchor="nw"
        )
        left.bind(
            "<Configure>",
            lambda _event: self.left_canvas.configure(
                scrollregion=self.left_canvas.bbox("all")
            ),
        )
        self.left_canvas.bind(
            "<Configure>",
            lambda event: self.left_canvas.itemconfigure(
                self._left_canvas_window, width=event.width
            ),
        )
        self.left_canvas.bind("<Enter>", self._bind_left_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_left_mousewheel)

        right = ttk.Frame(workspace, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        # Session card
        session_frame = self._make_card(left, "1 · Session metadata")
        session_frame.pack(fill="x", pady=(0, 12))
        for col in (1, 3, 5):
            session_frame.columnconfigure(col, weight=1)

        labels = [
            ("Participant", self.participant_var, 0, 0),
            ("Session", self.session_var, 0, 2),
            ("Run", self.run_var, 0, 4),
            ("Operator", self.operator_var, 2, 0),
            ("Site", self.site_var, 2, 2),
        ]
        for text, var, row, col in labels:
            ttk.Label(session_frame, text=text, style="App.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 6), pady=5
            )
            ttk.Entry(session_frame, textvariable=var, width=16).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=5
            )

        ttk.Label(session_frame, text="Experiment mode", style="App.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=5
        )
        self.experiment_mode_combo = ttk.Combobox(
            session_frame,
            textvariable=self.experiment_mode_var,
            values=[e.value for e in ExperimentMode],
            state="readonly",
            width=20,
        )
        self.experiment_mode_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=5)
        self.experiment_mode_combo.bind("<<ComboboxSelected>>", self._on_experiment_mode_changed)

        ttk.Label(session_frame, text="Notes", style="App.TLabel").grid(
            row=3, column=0, sticky="nw", padx=(0, 6), pady=(8, 5)
        )
        self.notes_text = tk.Text(
            session_frame,
            height=2,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            background=self.COLORS["surface_alt"],
            foreground=self.COLORS["text"],
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
        )
        self.notes_text.grid(row=3, column=1, columnspan=5, sticky="ew", pady=(8, 5))

        ttk.Button(
            session_frame,
            text="Apply metadata",
            command=self._apply_session_metadata,
            style="Secondary.TButton",
        ).grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="w")

        # Configuration card
        config_frame = self._make_card(left, "2 · Acquisition and decoder")
        config_frame.pack(fill="x", pady=(0, 12))
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Acquisition mode", style="App.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=6
        )
        self.mode_combo = ttk.Combobox(
            config_frame,
            textvariable=self.selected_mode_var,
            values=["EEG", "EMG", "IMU", "EEG+EMG", "EEG+IMU", "EMG+IMU", "EEG+EMG+IMU"],
            state="readonly",
            width=24,
        )
        self.mode_combo.grid(row=0, column=1, sticky="ew", pady=6)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        ttk.Label(config_frame, text="Decoder model", style="App.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=6
        )
        self.model_combo = ttk.Combobox(
            config_frame,
            textvariable=self.selected_model_var,
            values=[],
            state="readonly",
            width=32,
        )
        self.model_combo.grid(row=1, column=1, sticky="ew", pady=6)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        self.brainprint_label = ttk.Label(config_frame, text="", style="Muted.TLabel", wraplength=470)
        self.brainprint_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 2))

        # Decoder info mini-card
        self.decoder_info_frame = ttk.Frame(config_frame, style="Soft.TFrame", padding=10)
        self.decoder_info_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.decoder_info_frame.columnconfigure(1, weight=1)
        self.decoder_title_label = ttk.Label(
            self.decoder_info_frame, text="No decoder selected", style="Metric.TLabel"
        )
        self.decoder_title_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.decoder_meta_label = ttk.Label(
            self.decoder_info_frame,
            text="Select an online decoder to inspect its exported metadata.",
            style="MetricCaption.TLabel",
            wraplength=470,
            justify="left",
        )
        self.decoder_meta_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Action card
        actions_frame = self._make_card(left, "3 · Workflow")
        actions_frame.pack(fill="x", pady=(0, 12))
        for i in range(2):
            actions_frame.columnconfigure(i, weight=1)

        self.refresh_button = ttk.Button(
            actions_frame,
            text="Refresh streams",
            command=self._refresh_streams,
            style="Secondary.TButton",
        )
        self.refresh_button.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=5)

        self.validate_button = ttk.Button(
            actions_frame,
            text="Validate setup",
            command=self._validate_setup,
            style="Secondary.TButton",
        )
        self.validate_button.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=5)

        self.prepare_button = ttk.Button(
            actions_frame,
            text="Prepare run",
            command=self._prepare_run,
            style="Secondary.TButton",
        )
        self.prepare_button.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=5)

        self.launch_button = ttk.Button(
            actions_frame,
            text="Launch PsychoPy",
            command=self._launch_psychopy,
            style="Primary.TButton",
        )
        self.launch_button.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=5)

        # Quick summary card
        summary_frame = self._make_card(left, "System overview")
        summary_frame.pack(fill="both", expand=True)
        for i in range(3):
            summary_frame.columnconfigure(i, weight=1)

        self.metric_streams = self._create_metric(summary_frame, 0, "0", "streams")
        self.metric_modalities = self._create_metric(summary_frame, 1, "0", "modalities")
        self.metric_model = self._create_metric(summary_frame, 2, "—", "decoder")

        # Right-side tabbed view
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        streams_tab = ttk.Frame(notebook, style="Surface.TFrame", padding=12)
        validation_tab = ttk.Frame(notebook, style="Surface.TFrame", padding=12)
        prepared_tab = ttk.Frame(notebook, style="Surface.TFrame", padding=12)
        notebook.add(streams_tab, text="  Streams  ")
        notebook.add(validation_tab, text="  Validation  ")
        live_tab = ttk.Frame(notebook, style="Surface.TFrame", padding=12)
        notebook.add(prepared_tab, text="  Prepared run  ")
        notebook.add(live_tab, text="  Live monitor  ")

        streams_tab.rowconfigure(1, weight=1)
        streams_tab.columnconfigure(0, weight=1)
        streams_header = ttk.Frame(streams_tab, style="Surface.TFrame")
        streams_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        streams_header.columnconfigure(0, weight=1)
        ttk.Label(
            streams_header, text="Live LSL discovery", style="SectionTitle.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self.stream_indicator_frame = tk.Frame(streams_header, bg=self.COLORS["surface"])
        self.stream_indicator_frame.grid(row=0, column=1, sticky="e", padx=(8, 12))
        self.stream_indicator_labels = {}
        for name in ("EEG", "EMG", "IMU", "MARKERS"):
            label = tk.Label(
                self.stream_indicator_frame, text=f"○ {name}",
                bg=self.COLORS["surface"], fg=self.COLORS["muted"],
                font=("Segoe UI Semibold", 8), padx=5
            )
            label.pack(side="left")
            self.stream_indicator_labels[name] = label
        ttk.Button(
            streams_header,
            text="Refresh streams",
            command=self._refresh_streams,
            style="Primary.TButton",
        ).grid(row=0, column=2, sticky="e")
        self.streams_text = self._make_text_panel(streams_tab, height=25)
        self.streams_text.grid(row=1, column=0, sticky="nsew")

        validation_tab.rowconfigure(1, weight=1)
        validation_tab.columnconfigure(0, weight=1)
        ttk.Label(validation_tab, text="Configuration checks and runtime messages", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.validation_text = self._make_text_panel(validation_tab, height=25)
        self.validation_text.grid(row=1, column=0, sticky="nsew")

        prepared_tab.rowconfigure(1, weight=1)
        prepared_tab.columnconfigure(0, weight=1)
        ttk.Label(prepared_tab, text="Prepared bridge, files and services", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.prepared_text = self._make_text_panel(prepared_tab, height=25)
        self.prepared_text.grid(row=1, column=0, sticky="nsew")

        # Compact live monitor. It reads the decoder latest-result JSON and is
        # intentionally display-only, so it cannot affect experiment timing.
        live_tab.columnconfigure(0, weight=1)
        live_tab.rowconfigure(2, weight=1)
        live_header = ttk.Frame(live_tab, style="Surface.TFrame")
        live_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        live_header.columnconfigure(0, weight=1)
        ttk.Label(live_header, text="Online decoder monitor", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.live_status_label = tk.Label(
            live_header, text="WAITING", bg=self.COLORS["surface_alt"],
            fg=self.COLORS["muted"], font=("Segoe UI Semibold", 9), padx=10, pady=5
        )
        self.live_status_label.grid(row=0, column=1, sticky="e")

        live_cards = ttk.Frame(live_tab, style="Surface.TFrame")
        live_cards.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            live_cards.columnconfigure(col, weight=1)
        self.live_trial_value = self._create_live_card(live_cards, 0, "—", "TRIAL")
        self.live_target_value = self._create_live_card(live_cards, 1, "—", "TARGET")
        self.live_prediction_value = self._create_live_card(live_cards, 2, "—", "PREDICTION")
        self.live_confidence_value = self._create_live_card(live_cards, 3, "—", "CONFIDENCE")

        self.live_details_text = self._make_text_panel(live_tab, height=18)
        self.live_details_text.grid(row=2, column=0, sticky="nsew")
        self.live_details_text.insert("end", "NO LIVE RESULT YET\n", "warning")
        self.live_details_text.insert("end", "Launch an online experiment. This panel updates from latest.json and is most useful on a second screen.\n", "muted")

        footer = tk.Frame(outer, bg=self.COLORS["bg"])
        footer.pack(fill="x", pady=(10, 0))
        tk.Label(
            footer,
            text="TFM · Online multimodal silent and imagined speech decoding",
            bg=self.COLORS["bg"],
            fg=self.COLORS["muted"],
            font=("Segoe UI", 8),
        ).pack(side="left")
        tk.Label(
            footer,
            text="Stable research build",
            bg=self.COLORS["bg"],
            fg=self.COLORS["muted"],
            font=("Segoe UI", 8),
        ).pack(side="right")

    def _create_live_card(self, parent, column: int, value: str, caption: str) -> ttk.Label:
        card = tk.Frame(
            parent, bg=self.COLORS["surface_alt"],
            highlightthickness=1, highlightbackground=self.COLORS["border"],
            padx=10, pady=10,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 4, 0 if column == 3 else 4))
        value_label = tk.Label(
            card, text=value, bg=self.COLORS["surface_alt"], fg=self.COLORS["text"],
            font=("Segoe UI Semibold", 14), wraplength=150, justify="center"
        )
        value_label.pack(anchor="center")
        tk.Label(
            card, text=caption, bg=self.COLORS["surface_alt"], fg=self.COLORS["muted"],
            font=("Segoe UI", 8)
        ).pack(anchor="center", pady=(3, 0))
        return value_label

    def _create_metric(self, parent, column: int, value: str, caption: str) -> ttk.Label:
        card = ttk.Frame(parent, style="Soft.TFrame", padding=(10, 9))
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 4, 0 if column == 2 else 4))
        value_label = ttk.Label(card, text=value, style="Metric.TLabel")
        value_label.pack(anchor="center")
        ttk.Label(card, text=caption.upper(), style="MetricCaption.TLabel").pack(anchor="center", pady=(2, 0))
        return value_label

    def _bind_left_mousewheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_left_mousewheel)
        self.root.bind_all("<Button-4>", self._on_left_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_left_mousewheel_linux)

    def _unbind_left_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_left_mousewheel(self, event) -> None:
        if event.delta:
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_left_mousewheel_linux(self, event) -> None:
        direction = -1 if event.num == 4 else 1
        self.left_canvas.yview_scroll(direction, "units")

    # ------------------------------------------------------------------
    # Controller interactions
    # ------------------------------------------------------------------
    def _poll_background_tasks(self) -> None:
        result = self.controller.poll_psychopy_process()

        if result is not None:
            self.validation_text.delete("1.0", "end")
            return_code = result["return_code"]
            tag = "success" if return_code == 0 else "error"
            self.validation_text.insert(
                "end",
                f"PsychoPy finished with exit code {return_code}.\n",
                tag,
            )

            brainprint_result = result["brainprint_result"]
            if brainprint_result is None:
                self.validation_text.insert(
                    "end",
                    f"Brainprint step skipped: {result['brainprint_skipped_reason']}\n",
                    "muted",
                )
            else:
                bp_tag = "success" if str(brainprint_result.status).lower() in {"ok", "success", "completed"} else "info"
                self.validation_text.insert("end", f"Brainprint status: {brainprint_result.status}\n", bp_tag)
                self.validation_text.insert("end", f"{brainprint_result.message}\n")

                if brainprint_result.candidate_npz:
                    self.validation_text.insert("end", f"Candidate NPZ: {brainprint_result.candidate_npz}\n", "muted")
                if brainprint_result.comparison_json:
                    self.validation_text.insert("end", f"Comparison JSON: {brainprint_result.comparison_json}\n", "muted")
                if brainprint_result.best_match_file and brainprint_result.best_match_cosine is not None:
                    self.validation_text.insert("end", f"Best match: {brainprint_result.best_match_file}\n")
                    self.validation_text.insert("end", f"Best cosine: {brainprint_result.best_match_cosine:.4f}\n")

            self._render_state()

        self.root.after(1000, self._poll_background_tasks)

    def _poll_live_decoder_result(self) -> None:
        state = self.controller.get_state()
        latest_path = state.online_decoder_latest_path
        if latest_path:
            try:
                path = Path(latest_path)
                if path.is_file():
                    stat = path.stat()
                    signature = (str(path), stat.st_mtime_ns, stat.st_size)
                    if signature != self._last_live_result_signature:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        self._last_live_result_signature = signature
                        self._render_live_result(payload, path)
            except (OSError, PermissionError, json.JSONDecodeError, ValueError) as exc:
                # latest.json may be briefly replaced while the decoder writes it.
                # Keep the last valid display and retry on the next poll.
                self.live_status_label.config(
                    text="UPDATING", fg=self.COLORS["warning"], bg=self.COLORS["warning_soft"]
                )
        elif not state.online_decoder_running:
            self.live_status_label.config(
                text="WAITING", fg=self.COLORS["muted"], bg=self.COLORS["surface_alt"]
            )
        self.root.after(400, self._poll_live_decoder_result)

    def _render_live_result(self, payload: dict, path: Path) -> None:
        trial = payload.get("trial_number", payload.get("trial", payload.get("trial_index", "—")))
        target = payload.get("target_word", payload.get("target_label", payload.get("target", "—")))
        prediction = payload.get("predicted_word", payload.get("predicted_label", payload.get("prediction", "—")))
        confidence = payload.get("confidence")
        status = str(payload.get("status", "ok")).lower()

        self.live_trial_value.config(text=str(trial))
        self.live_target_value.config(text=str(target))
        self.live_prediction_value.config(text=str(prediction))
        if isinstance(confidence, (int, float)):
            self.live_confidence_value.config(text=f"{100.0 * float(confidence):.1f}%")
        else:
            self.live_confidence_value.config(text="—")

        healthy = status in {"ok", "success", "completed", "ready"}
        self.live_status_label.config(
            text="LIVE" if healthy else status.upper(),
            fg=self.COLORS["success"] if healthy else self.COLORS["error"],
            bg=self.COLORS["success_soft"] if healthy else self.COLORS["error_soft"],
        )

        self.live_details_text.delete("1.0", "end")
        self.live_details_text.insert("end", "LATEST DECODER RESULT\n", "success" if healthy else "error")
        rows = [
            ("Trial", trial),
            ("Target", target),
            ("Prediction", prediction),
            ("Confidence", f"{float(confidence):.4f}" if isinstance(confidence, (int, float)) else "—"),
            ("Backend", payload.get("model_backend", payload.get("backend", "—"))),
            ("Status", payload.get("status", "—")),
            ("Error", payload.get("error_message") or "none"),
            ("File", path),
        ]
        for label, value in rows:
            self.live_details_text.insert("end", f"{label}\n", "heading")
            self.live_details_text.insert("end", f"{value}\n\n", "muted")

    def _refresh_streams(self) -> None:
        self.controller.refresh_streams()
        self._render_state()

    def _apply_session_metadata(self) -> None:
        notes = self.notes_text.get("1.0", "end").strip()
        self.controller.set_session_metadata(
            participant_id=self.participant_var.get(),
            session_id=self.session_var.get(),
            run_id=self.run_var.get(),
            experiment_mode=self.experiment_mode_var.get(),
            notes=notes,
            operator=self.operator_var.get(),
            site=self.site_var.get(),
        )
        self._render_state()
        messagebox.showinfo("Session metadata", "Session metadata updated successfully.")

    def _on_experiment_mode_changed(self, event=None) -> None:
        mode_name = self.experiment_mode_var.get()
        if mode_name:
            self.controller.set_experiment_mode(mode_name)
            self._render_state()
            self.left_canvas.yview_moveto(0.0)

    def _on_mode_changed(self, event=None) -> None:
        mode_name = self.selected_mode_var.get()
        if mode_name:
            self.controller.set_selected_mode(mode_name)
            self._render_state()

    def _on_model_changed(self, event=None) -> None:
        model_value = self.selected_model_var.get()
        model_key = get_model_key_from_display_name(model_value)
        if model_key:
            self.controller.set_selected_model(model_key)
            self._render_state()

    def _validate_setup(self) -> None:
        errors = self.controller.validate_configuration()
        self.validation_text.delete("1.0", "end")
        if not errors:
            self.validation_text.insert("end", "READY\n", "success")
            self.validation_text.insert("end", "Configuration looks valid and is ready to prepare.\n")
        else:
            self.validation_text.insert("end", "VALIDATION ISSUES\n", "error")
            for err in errors:
                self.validation_text.insert("end", f"• {err}\n", "error")
        self._render_state()

    def _prepare_run(self) -> None:
        errors = self.controller.prepare_run()
        self.validation_text.delete("1.0", "end")

        if errors:
            self.validation_text.insert("end", "PREPARATION FAILED\n", "error")
            for err in errors:
                self.validation_text.insert("end", f"• {err}\n", "error")
            self._render_state()
            return

        self.validation_text.insert("end", "RUN PREPARED\n", "success")
        self.validation_text.insert("end", "Bridge, metadata and manifest files were created successfully.\n")
        self._render_state()
        messagebox.showinfo(
            "Run prepared",
            "Run files were created successfully.\n\nThe GUI bridge is ready and PsychoPy can now be launched.",
        )

    def _launch_psychopy(self) -> None:
        self.validation_text.delete("1.0", "end")
        errors = self.controller.launch_psychopy()

        if errors:
            self.validation_text.insert("end", "LAUNCH FAILED\n", "error")
            for err in errors:
                self.validation_text.insert("end", f"• {err}\n", "error")
            self._render_state()
            return

        self.validation_text.insert("end", "PSYCHOPY LAUNCHED\n", "success")
        if self.controller.get_state().online_decoder_running:
            self.validation_text.insert("end", "Online decoder service launched successfully.\n", "info")
        self.validation_text.insert("end", "Waiting for the experiment process to finish...\n", "muted")
        self._render_state()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_state(self) -> None:
        state = self.controller.get_state()

        badge_text, fg_key, bg_key = self.STATUS_STYLE.get(
            state.status, (state.status.value, "muted", "surface_alt")
        )
        self.status_badge.config(
            text=badge_text,
            fg=self.COLORS[fg_key],
            bg=self.COLORS[bg_key],
        )
        self.error_label.config(text=state.error_message or "")

        self.experiment_mode_combo["values"] = state.available_experiment_modes
        self.mode_combo["values"] = (
            state.available_modes
            if state.available_modes
            else ["EEG", "EMG", "IMU", "EEG+EMG", "EEG+IMU", "EMG+IMU", "EEG+EMG+IMU"]
        )
        self.model_combo["values"] = (
            [get_model_display_name(model_key) for model_key in state.available_models]
            if state.available_models
            else []
        )

        self.experiment_mode_var.set(state.selected_experiment_mode or "")
        self.selected_mode_var.set(state.selected_mode or "")
        self.selected_model_var.set(get_model_display_name(state.selected_model))

        if state.selected_experiment_mode == ExperimentMode.ONLINE_IMAGINED.value:
            self.mode_combo.configure(state="disabled")
            self.model_combo.configure(state="readonly")
            selected_name = get_model_display_name(state.selected_model) or "No compatible imagined decoder"
            self.brainprint_label.config(text=f"Imagined mode · EEG only · {selected_name}")
        elif state.selected_experiment_mode == ExperimentMode.RECORDING.value:
            self.mode_combo.configure(state="readonly")
            self.model_combo.configure(state="disabled")
            self.brainprint_label.config(text="Recording mode · automatic post-recording brainprint processing")
        else:
            self.mode_combo.configure(state="readonly")
            self.model_combo.configure(state="readonly")
            self.brainprint_label.config(text="Online silent mode · optional EEG reliability scaling when available")

        self._render_decoder_info(state.selected_model)
        self._render_streams(state)
        self._render_prepared_run(state)

        detected_modalities = sorted(m.value for m in state.detected_modalities)
        self.metric_streams.config(text=str(len(state.detected_streams)))
        self.metric_modalities.config(text=str(len(detected_modalities)))
        model_text = "—"
        if state.selected_model:
            try:
                model_text = "+".join(sorted(m.value for m in get_model_spec(state.selected_model).required_modalities))
            except Exception:
                model_text = "set"
        self.metric_model.config(text=model_text)

        if self.validation_text.get("1.0", "end").strip() == "":
            self.validation_text.insert("end", "NEXT STEP\n", "info")
            self.validation_text.insert(
                "end",
                "Validate the configuration, then prepare and launch the experiment.\n",
                "muted",
            )

    def _render_decoder_info(self, model_key: str | None) -> None:
        if not model_key:
            self.decoder_title_label.config(text="No decoder selected")
            self.decoder_meta_label.config(
                text="Recording mode does not require an online decoder, or no compatible model is currently selected."
            )
            return

        try:
            spec = get_model_spec(model_key)
        except Exception:
            self.decoder_title_label.config(text=get_model_display_name(model_key))
            self.decoder_meta_label.config(text="Decoder metadata could not be loaded from the manifest.")
            return

        modalities = " + ".join(sorted(m.value for m in spec.required_modalities))
        channels = ", ".join(
            f"{key.upper()} {value} ch" for key, value in (spec.input_channels or {}).items()
        ) or "not specified"
        window = "not specified"
        if spec.window_start_s is not None and spec.window_end_s is not None:
            window = f"{spec.window_start_s:+.2f} to {spec.window_end_s:+.2f} s"
        fusion = spec.fusion_mode or "single modality"

        self.decoder_title_label.config(text=spec.display_name)
        detail = f"Task: {spec.target_task or '—'}   ·   Modalities: {modalities}\n"
        detail += f"Input: {channels}   ·   Window: {window}   ·   Fusion: {fusion}"
        if spec.description:
            detail += f"\n{spec.description}"
        self.decoder_meta_label.config(text=detail)

    def _render_streams(self, state) -> None:
        detected_names = {
            (stream.modality.value if stream.modality else "OTHER").upper()
            for stream in state.detected_streams
        }
        for name, label in self.stream_indicator_labels.items():
            connected = name in detected_names or (name == "MARKERS" and "MARKER" in detected_names)
            label.config(
                text=f"● {name}" if connected else f"○ {name}",
                fg=self.COLORS["success"] if connected else self.COLORS["muted"],
            )

        self.streams_text.delete("1.0", "end")
        if not state.detected_streams:
            self.streams_text.insert("end", "NO STREAMS DETECTED\n", "warning")
            self.streams_text.insert(
                "end",
                "Start the acquisition device or a test publisher, then press Refresh streams.\n",
                "muted",
            )
            return

        by_modality: dict[str, list] = {}
        for stream in state.detected_streams:
            modality = stream.modality.value if stream.modality else "OTHER"
            by_modality.setdefault(modality, []).append(stream)

        preferred_order = ["EEG", "EMG", "IMU", "MARKERS", "OTHER"]
        for modality in preferred_order:
            streams = by_modality.get(modality, [])
            if not streams:
                continue
            self.streams_text.insert("end", f"{modality}  ·  {len(streams)} stream(s)\n", "heading")
            for stream in streams:
                self.streams_text.insert("end", "● ", "success")
                self.streams_text.insert("end", f"{stream.name}\n", "heading")
                self.streams_text.insert(
                    "end",
                    f"   {stream.channel_count} channels · {stream.nominal_srate:g} Hz",
                    "muted",
                )
                if stream.hostname:
                    self.streams_text.insert("end", f" · host {stream.hostname}", "muted")
                self.streams_text.insert("end", "\n")
                if stream.source_id:
                    self.streams_text.insert("end", f"   source: {stream.source_id}\n", "muted")
            self.streams_text.insert("end", "\n")

    def _render_prepared_run(self, state) -> None:
        self.prepared_text.delete("1.0", "end")
        if not state.prepared_run_basename:
            self.prepared_text.insert("end", "NO RUN PREPARED\n", "warning")
            self.prepared_text.insert(
                "end",
                "Apply session metadata, validate the setup and press Prepare run.\n",
                "muted",
            )
            return

        rows = [
            ("Experiment", state.selected_experiment_mode),
            ("Acquisition", state.selected_mode),
            ("Decoder", get_model_display_name(state.selected_model) or "N/A for recording"),
            ("Basename", state.prepared_run_basename),
            ("Run directory", state.prepared_run_dir),
            ("Calibration XDF", state.prepared_xdf_path),
            ("Manifest", state.prepared_manifest_path),
            ("Metadata", state.prepared_metadata_path),
            ("Bridge", state.prepared_bridge_path),
            ("PsychoPy", state.psychopy_script_path),
            ("Bundle", state.prepared_decoder_bundle_path or "ensemble / none"),
            ("Online results", state.prepared_online_results_dir or "none"),
            ("Decoder running", str(state.online_decoder_running)),
        ]

        self.prepared_text.insert("end", "RUN READY\n", "success")
        for label, value in rows:
            self.prepared_text.insert("end", f"{label}\n", "heading")
            self.prepared_text.insert("end", f"{value or '—'}\n\n", "muted")

        if state.online_decoder_service_status_path:
            self.prepared_text.insert("end", "Service status JSON\n", "heading")
            self.prepared_text.insert("end", f"{state.online_decoder_service_status_path}\n\n", "muted")
        if state.online_decoder_latest_path:
            self.prepared_text.insert("end", "Latest result JSON\n", "heading")
            self.prepared_text.insert("end", f"{state.online_decoder_latest_path}\n\n", "muted")
        if state.online_decoder_summary_path:
            self.prepared_text.insert("end", "Run summary JSON\n", "heading")
            self.prepared_text.insert("end", f"{state.online_decoder_summary_path}\n\n", "muted")
        if state.last_pipeline_message:
            self.prepared_text.insert("end", "Last brainprint result\n", "heading")
            self.prepared_text.insert("end", state.last_pipeline_message + "\n", "muted")


def main() -> None:
    root = tk.Tk()
    TFMGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

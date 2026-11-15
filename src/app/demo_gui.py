"""Lightweight Tkinter demo UI (Tk ships with the python.org Windows installer - no extra dependency).

Flow: Open file -> Run -> switch tabs (Detections / Lanes / Depth / Combined / All four) -> optionally Save.
All processing is done by src.app.session / src.perception (the GUI contains no perception logic).
"""
from __future__ import annotations

import base64
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2

from src.app.session import OutputWriter, build_pipeline_for, run_session
from src.perception.render import mosaic
from src.utils.config import apply_set_overrides, load_config, resolve_path

TABS = ["combined", "detections", "lanes", "depth", "all four"]


def to_photo(img_bgr, max_w: int, max_h: int) -> tk.PhotoImage:
    h, w = img_bgr.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img_bgr)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class DemoApp:
    def __init__(self, root: tk.Tk, config: Optional[str] = None):
        self.root = root
        root.title("AIML-04 Autonomous Driving Perception Stack - Demo")
        root.geometry("1100x760")
        self.cfg_path = config
        self.source: Optional[str] = None
        self.q: "queue.Queue" = queue.Queue(maxsize=2)
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.latest = None
        self.photo = None

        top = ttk.Frame(root, padding=6); top.pack(fill="x")
        ttk.Button(top, text="Open image / video...", command=self.open_file).pack(side="left")
        self.lbl = ttk.Label(top, text="no file selected"); self.lbl.pack(side="left", padx=8)
        ttk.Label(top, text="Detector:").pack(side="left", padx=(16, 2))
        self.det = tk.StringVar(value="yolo")
        ttk.Combobox(top, textvariable=self.det, values=["yolo", "onnx", "groundtruth"], width=12, state="readonly").pack(side="left")
        self.use_depth = tk.BooleanVar(value=True); self.use_lanes = tk.BooleanVar(value=True); self.use_seg = tk.BooleanVar(value=False)
        self.save = tk.BooleanVar(value=True)
        for text, var in (("Depth (MiDaS)", self.use_depth), ("Lanes", self.use_lanes), ("Seg. lanes", self.use_seg), ("Save output", self.save)):
            ttk.Checkbutton(top, text=text, variable=var).pack(side="left", padx=4)
        self.run_btn = ttk.Button(top, text="Run", command=self.run); self.run_btn.pack(side="right", padx=2)
        ttk.Button(top, text="Stop", command=self.stop).pack(side="right")

        self.nb = ttk.Notebook(root); self.nb.pack(fill="both", expand=True, padx=6)
        self.tab_frames = {}
        for t in TABS:
            f = ttk.Frame(self.nb); self.nb.add(f, text=t.title())
            lab = ttk.Label(f, anchor="center"); lab.pack(fill="both", expand=True)
            self.tab_frames[t] = lab
        self.status = tk.StringVar(value="Ready. Open a driving image or video, then press Run.")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w", padding=4).pack(fill="x", side="bottom")
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.show())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self.poll)

    def open_file(self):
        p = filedialog.askopenfilename(initialdir=str(resolve_path("datasets/samples")), title="Choose image or video",
                                       filetypes=[("Images / videos", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if p:
            self.source = p
            self.lbl.config(text=Path(p).name)

    def run(self):
        if not self.source:
            messagebox.showinfo("No input", "Please open an image or video first.")
            return
        if self.thread and self.thread.is_alive():
            return
        self.stop_flag.clear()
        self.run_btn.state(["disabled"])
        self.status.set("Loading models (first run can take a while: downloads + warm-up)...")
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()

    def worker(self):
        writer = None
        try:
            cfg = load_config(self.cfg_path)
            cfg["detector"]["backend"] = self.det.get()
            cfg["pipeline"].update(depth=self.use_depth.get(), lanes=self.use_lanes.get(), lane_seg=self.use_seg.get())
            gt = None
            if self.det.get() == "groundtruth":
                cand = Path(self.source).with_name(Path(self.source).stem + "_gt.json")
                gt = str(cand) if cand.exists() else None
            pipe, kind, fps, _ = build_pipeline_for(self.source, cfg, gt)
            if self.save.get():
                writer = OutputWriter(cfg["run"]["output_dir"], Path(self.source).stem + "_gui", kind, fps)
            t0, n = time.perf_counter(), 0
            for idx, frame, res, panels in run_session(self.source, cfg, pipeline=pipe):
                if self.stop_flag.is_set():
                    break
                if writer:
                    writer.write(panels, "mosaic", idx)
                n += 1
                info = f"frame {idx} | {len(res.tracks)} tracks | {res.decision.state} | {res.timings_ms['total']:.0f} ms | {n / (time.perf_counter() - t0):.1f} fps"
                try:
                    self.q.put_nowait((panels, info))
                except queue.Full:
                    pass
            msg = f"Finished {n} frame(s)." + (f" Saved: {writer.path}" if writer and writer.path else "")
            self.q.put(({}, msg))
        except Exception as e:  # show any error in the UI instead of dying silently
            self.q.put(({}, f"ERROR: {e}"))
        finally:
            if writer:
                writer.close()
            self.q.put(("done", ""))

    def poll(self):
        try:
            while True:
                panels, info = self.q.get_nowait()
                if panels == "done":
                    self.run_btn.state(["!disabled"])
                    continue
                self.status.set(info)
                if panels:
                    self.latest = panels
                    self.show()
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def show(self):
        if not self.latest:
            return
        name = TABS[self.nb.index(self.nb.select())]
        img = mosaic(self.latest) if name == "all four" else self.latest[name]
        w = max(self.tab_frames[name].winfo_width(), 480)    # winfo_* returns 1 before the window is mapped
        h = max(self.tab_frames[name].winfo_height(), 320)
        self.photo = to_photo(img, w, h)
        self.tab_frames[name].config(image=self.photo)

    def on_close(self):
        self.stop_flag.set()
        self.root.destroy()


def launch(config: Optional[str] = None):
    root = tk.Tk()
    DemoApp(root, config)
    root.mainloop()

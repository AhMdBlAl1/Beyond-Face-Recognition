import tkinter as tk
from tkinter import ttk, messagebox, font
import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime
import threading
from PIL import Image, ImageTk
import csv

# ─── Files ───────────────────────────────────────────────────────────────────
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# ─── Color Palette ────────────────────────────────────────────────────────────
BG        = "#0D0F14"
CARD      = "#161A23"
ACCENT    = "#00E5FF"
ACCENT2   = "#7C3AED"
SUCCESS   = "#22C55E"
DANGER    = "#EF4444"
TEXT      = "#F0F4FF"
MUTED     = "#6B7280"
BORDER    = "#1F2937"

# ═══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data.get("encodings", []), data.get("names", [])
    return [], []


def save_faces(encodings, names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")


def mark_attendance(name):
    ensure_attendance_file()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "r+") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0] == name and parts[1] == date:
                return False          # already logged today
        f.write(f"{name},{date},{time}\n")
    return True


def load_attendance():
    ensure_attendance_file()
    rows = []
    with open(ATTENDANCE_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
#  App
# ═══════════════════════════════════════════════════════════════════════════════

class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FaceID — Attendance System")
        self.geometry("1100x720")
        self.configure(bg=BG)
        self.resizable(True, True)

        # state
        self.known_encodings, self.known_names = load_faces()
        self.camera_running    = False
        self.camera_thread     = None
        self.cap               = None
        self.current_frame     = None
        self.mode              = None   # "register" | "recognize"

        # register temp state
        self.reg_name          = tk.StringVar()
        self.reg_captured_enc  = None
        self.reg_edit_index    = None   # index when editing existing person

        self._build_ui()

    # ─── UI Shell ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Sidebar ──
        self.sidebar = tk.Frame(self, bg=CARD, width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="FACE ID", bg=CARD, fg=ACCENT,
                 font=("Courier New", 18, "bold")).pack(pady=(32, 4))
        tk.Label(self.sidebar, text="Attendance System", bg=CARD, fg=MUTED,
                 font=("Courier New", 9)).pack(pady=(0, 32))

        self._nav_btn("＋  New Register",  self._show_register)
        self._nav_btn("▶  Start Recognition", self._show_recognize)
        self._nav_btn("📋  Attendance Log",   self._show_attendance)
        self._nav_btn("👥  Registered Users", self._show_users)

        # version tag at bottom
        tk.Label(self.sidebar, text="v1.0", bg=CARD, fg=BORDER,
                 font=("Courier New", 8)).pack(side="bottom", pady=12)

        # ── Main area ──
        self.main = tk.Frame(self, bg=BG)
        self.main.pack(side="left", fill="both", expand=True)

        self._show_home()

    def _nav_btn(self, label, cmd):
        btn = tk.Button(
            self.sidebar, text=label, command=cmd,
            bg=CARD, fg=TEXT, activebackground=BORDER, activeforeground=ACCENT,
            relief="flat", anchor="w", padx=20, pady=12,
            font=("Courier New", 10), cursor="hand2", bd=0
        )
        btn.pack(fill="x", padx=8, pady=2)
        btn.bind("<Enter>", lambda e: btn.config(bg=BORDER))
        btn.bind("<Leave>", lambda e: btn.config(bg=CARD))

    # ─── clear main ───────────────────────────────────────────────────────────

    def _clear_main(self):
        self._stop_camera()
        for w in self.main.winfo_children():
            w.destroy()

    def _page_title(self, title, subtitle=""):
        tk.Label(self.main, text=title, bg=BG, fg=TEXT,
                 font=("Courier New", 22, "bold")).pack(anchor="w", padx=40, pady=(36, 2))
        if subtitle:
            tk.Label(self.main, text=subtitle, bg=BG, fg=MUTED,
                     font=("Courier New", 10)).pack(anchor="w", padx=40, pady=(0, 24))

    # ─── Home ─────────────────────────────────────────────────────────────────

    def _show_home(self):
        self._clear_main()
        self._page_title("Welcome back.", "Select an action from the sidebar.")

        cards = [
            ("＋", "Register",   "Add a new person\nto the system",    self._show_register,    ACCENT),
            ("▶", "Recognise",   "Start live face\nrecognition",        self._show_recognize,   SUCCESS),
            ("📋", "Attendance",  "View attendance\nlog",               self._show_attendance,  ACCENT2),
            ("👥", "Users",       "Manage registered\nprofiles",        self._show_users,       "#F59E0B"),
        ]

        row = tk.Frame(self.main, bg=BG)
        row.pack(padx=40, pady=20)
        for icon, title, desc, cmd, clr in cards:
            c = tk.Frame(row, bg=CARD, width=200, height=160,
                         highlightbackground=BORDER, highlightthickness=1, cursor="hand2")
            c.pack(side="left", padx=10)
            c.pack_propagate(False)
            tk.Label(c, text=icon, bg=CARD, fg=clr,
                     font=("Courier New", 28)).pack(pady=(24, 4))
            tk.Label(c, text=title, bg=CARD, fg=TEXT,
                     font=("Courier New", 12, "bold")).pack()
            tk.Label(c, text=desc, bg=CARD, fg=MUTED,
                     font=("Courier New", 9), justify="center").pack(pady=(4, 0))
            c.bind("<Button-1>", lambda e, f=cmd: f())
            for child in c.winfo_children():
                child.bind("<Button-1>", lambda e, f=cmd: f())

        # stats
        n_users = len(set(self.known_names))
        rows = load_attendance()
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for r in rows if r.get("Date") == today)

        sf = tk.Frame(self.main, bg=BG)
        sf.pack(padx=40, pady=10, anchor="w")
        self._stat_chip(sf, str(n_users), "Registered")
        self._stat_chip(sf, str(len(rows)), "Total Logs")
        self._stat_chip(sf, str(today_count), "Today")

    def _stat_chip(self, parent, val, label):
        f = tk.Frame(parent, bg=BORDER)
        f.pack(side="left", padx=6, pady=4)
        tk.Label(f, text=f"  {val}  ", bg=BORDER, fg=ACCENT,
                 font=("Courier New", 18, "bold")).pack(side="left")
        tk.Label(f, text=f"{label}  ", bg=BORDER, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left")

    # ─── Register ─────────────────────────────────────────────────────────────

    def _show_register(self, edit_index=None):
        self._clear_main()
        self.mode           = "register"
        self.reg_captured_enc = None
        self.reg_edit_index = edit_index

        if edit_index is not None:
            self.reg_name.set(self.known_names[edit_index])
            title    = "Edit Profile"
            subtitle = "Update name or re-capture face photo."
        else:
            self.reg_name.set("")
            title    = "New Register"
            subtitle = "Capture your face and enter your name."

        self._page_title(title, subtitle)

        content = tk.Frame(self.main, bg=BG)
        content.pack(fill="both", expand=True, padx=40)

        # left – camera feed
        left = tk.Frame(content, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=4)

        self.cam_label = tk.Label(left, bg="#000", text="Camera loading…",
                                  fg=MUTED, font=("Courier New", 11))
        self.cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        # right – controls
        right = tk.Frame(content, bg=BG, width=280)
        right.pack(side="left", fill="y", pady=4)
        right.pack_propagate(False)

        tk.Label(right, text="Full Name", bg=BG, fg=MUTED,
                 font=("Courier New", 9)).pack(anchor="w", pady=(8, 2))
        self.name_entry = tk.Entry(
            right, textvariable=self.reg_name,
            bg=CARD, fg=TEXT, insertbackground=ACCENT,
            relief="flat", font=("Courier New", 13), bd=0
        )
        self.name_entry.pack(fill="x", ipady=10, padx=2)
        tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", padx=2)

        tk.Frame(right, bg=BG, height=20).pack()

        self.capture_btn = self._action_btn(
            right, "📸  Capture Face", self._capture_face, ACCENT)
        self.capture_btn.pack(fill="x", pady=4)

        self.status_lbl = tk.Label(right, text="", bg=BG, fg=MUTED,
                                   font=("Courier New", 9), wraplength=260,
                                   justify="left")
        self.status_lbl.pack(anchor="w", pady=4)

        self.save_btn = self._action_btn(
            right, "💾  Save Profile", self._save_registration, SUCCESS)
        self.save_btn.pack(fill="x", pady=4)
        self.save_btn.config(state="disabled")

        self._action_btn(right, "✖  Cancel", self._show_home, DANGER).pack(fill="x", pady=4)

        self._start_camera()

    def _capture_face(self):
        if self.current_frame is None:
            self.status_lbl.config(text="⚠ Camera not ready.", fg=DANGER)
            return
        frame = self.current_frame.copy()
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb)
        encs  = face_recognition.face_encodings(rgb, locs)
        if len(encs) == 0:
            self.status_lbl.config(text="No face detected. Try again.", fg=DANGER)
            return
        self.reg_captured_enc = encs[0]
        self.status_lbl.config(text="✔ Face captured!", fg=SUCCESS)
        self.save_btn.config(state="normal")

    def _save_registration(self):
        name = self.reg_name.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a name.")
            return
        if self.reg_captured_enc is None and self.reg_edit_index is None:
            messagebox.showwarning("No face", "Please capture your face first.")
            return

        if self.reg_edit_index is not None:
            # editing existing
            self.known_names[self.reg_edit_index] = name
            if self.reg_captured_enc is not None:
                self.known_encodings[self.reg_edit_index] = self.reg_captured_enc
        else:
            self.known_encodings.append(self.reg_captured_enc)
            self.known_names.append(name)

        save_faces(self.known_encodings, self.known_names)
        messagebox.showinfo("Saved", f"'{name}' registered successfully!")
        self._show_home()

    # ─── Recognition ──────────────────────────────────────────────────────────

    def _show_recognize(self):
        self._clear_main()
        self.mode = "recognize"

        self._page_title("Live Recognition", "Face detection & attendance logging.")

        content = tk.Frame(self.main, bg=BG)
        content.pack(fill="both", expand=True, padx=40)

        left = tk.Frame(content, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=4)

        self.cam_label = tk.Label(left, bg="#000", text="Starting camera…",
                                  fg=MUTED, font=("Courier New", 11))
        self.cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        right = tk.Frame(content, bg=BG, width=280)
        right.pack(side="left", fill="y", pady=4)
        right.pack_propagate(False)

        tk.Label(right, text="Detected", bg=BG, fg=MUTED,
                 font=("Courier New", 9)).pack(anchor="w", pady=(8, 4))

        self.detect_lbl = tk.Label(right, text="—", bg=BG, fg=ACCENT,
                                   font=("Courier New", 16, "bold"),
                                   wraplength=260, justify="left")
        self.detect_lbl.pack(anchor="w")

        self.log_lbl = tk.Label(right, text="", bg=BG, fg=SUCCESS,
                                font=("Courier New", 9), wraplength=260,
                                justify="left")
        self.log_lbl.pack(anchor="w", pady=4)

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=12)

        self._action_btn(right, "📋  View Attendance", self._show_attendance, ACCENT2).pack(fill="x", pady=4)
        self._action_btn(right, "✖  Stop & Exit", self._show_home, DANGER).pack(fill="x", pady=4)

        self._start_camera()

    # ─── Attendance Log ───────────────────────────────────────────────────────

    def _show_attendance(self):
        self._clear_main()
        self._page_title("Attendance Log", "History of all recorded entries.")

        rows = load_attendance()

        # Filter bar
        bar = tk.Frame(self.main, bg=BG)
        bar.pack(padx=40, pady=(0, 12), anchor="w")
        tk.Label(bar, text="Filter by name:", bg=BG, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left")
        self.filter_var = tk.StringVar()
        fe = tk.Entry(bar, textvariable=self.filter_var, bg=CARD, fg=TEXT,
                      insertbackground=ACCENT, relief="flat",
                      font=("Courier New", 11), width=20, bd=0)
        fe.pack(side="left", ipady=6, padx=(8, 0))
        tk.Frame(bar, bg=ACCENT, height=1).pack(side="left")  # cosmetic

        self.filter_var.trace_add("write", lambda *a: self._refresh_table(rows))

        # table frame
        tf = tk.Frame(self.main, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        tf.pack(padx=40, fill="both", expand=True, pady=(0, 24))

        cols = ("Name", "Date", "Time")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=20)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                         background=CARD, foreground=TEXT,
                         fieldbackground=CARD, rowheight=30,
                         font=("Courier New", 10))
        style.configure("Treeview.Heading",
                         background=BORDER, foreground=ACCENT,
                         font=("Courier New", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT2)])

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=200, anchor="w")

        sb = tk.Scrollbar(tf, orient="vertical", command=self.tree.yview,
                          bg=BORDER, troughcolor=CARD)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)

        self._all_rows = rows
        self._refresh_table(rows)

    def _refresh_table(self, rows=None):
        if rows is None:
            rows = self._all_rows
        q = self.filter_var.get().strip().lower()
        filtered = [r for r in rows if q in r.get("Name", "").lower()] if q else rows
        self.tree.delete(*self.tree.get_children())
        for r in reversed(filtered):
            self.tree.insert("", "end", values=(r.get("Name"), r.get("Date"), r.get("Time")))

    # ─── Registered Users ─────────────────────────────────────────────────────

    def _show_users(self):
        self._clear_main()
        self._page_title("Registered Users", "Edit name or re-capture face for any profile.")

        container = tk.Frame(self.main, bg=BG)
        container.pack(fill="both", expand=True, padx=40, pady=8)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(container, orient="vertical", command=canvas.yview,
                          bg=BORDER, troughcolor=BG)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        if not self.known_names:
            tk.Label(inner, text="No users registered yet.", bg=BG, fg=MUTED,
                     font=("Courier New", 12)).pack(pady=40)
            return

        seen = {}
        for i, name in enumerate(self.known_names):
            seen[name] = i  # keep last index per name

        for name, idx in seen.items():
            row = tk.Frame(inner, bg=CARD,
                           highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", pady=4)

            tk.Label(row, text="👤", bg=CARD, font=("Courier New", 20)).pack(side="left", padx=16, pady=10)
            tk.Label(row, text=name, bg=CARD, fg=TEXT,
                     font=("Courier New", 13, "bold")).pack(side="left", padx=4)

            def _edit(i=idx):
                self._show_register(edit_index=i)

            def _delete(n=name):
                if messagebox.askyesno("Delete", f"Remove '{n}' from system?"):
                    indices = [j for j, x in enumerate(self.known_names) if x == n]
                    for j in sorted(indices, reverse=True):
                        del self.known_encodings[j]
                        del self.known_names[j]
                    save_faces(self.known_encodings, self.known_names)
                    self._show_users()

            self._action_btn(row, "✏  Edit", _edit, ACCENT).pack(side="right", padx=8, pady=8)
            self._action_btn(row, "🗑  Delete", _delete, DANGER).pack(side="right", padx=4, pady=8)

    # ─── Camera helpers ───────────────────────────────────────────────────────

    def _start_camera(self):
        self.camera_running = True
        self.cap = cv2.VideoCapture(0)
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _camera_loop(self):
        process_this_frame = True 
        while self.camera_running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Mirror Effect
            frame = cv2.flip(frame, 1) 
            self.current_frame = frame.copy()
            display = frame.copy()

            if self.mode == "recognize" and self.known_encodings:
                # 2. Speed Hack: Resize to 25%
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                if process_this_frame:
                    locs = face_recognition.face_locations(rgb_small)
                    encs = face_recognition.face_encodings(rgb_small, locs)
                    
                    names_found = []
                    for enc, loc in zip(encs, locs):
                        matches = face_recognition.compare_faces(self.known_encodings, enc)
                        name = "Unknown"
                        if True in matches:
                            # استخدام الـ Distance لأعلى دقة
                            distances = face_recognition.face_distance(self.known_encodings, enc)
                            best_idx = np.argmin(distances)
                            name = self.known_names[best_idx]
                            
                            if mark_attendance(name):
                                self.after(0, lambda n=name: self.log_lbl.config(
                                    text=f"✔ {n} logged at {datetime.now().strftime('%H:%M:%S')}"))
                        names_found.append(name)

                process_this_frame = not process_this_frame

                # 3. Draw Boxes (Scale back up x4)
                for (top, right, bottom, left), name in zip(locs, names_found):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    color = (0, 229, 255) if name != "Unknown" else (239, 68, 68)
                    cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                    cv2.putText(display, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if names_found:
                    self.after(0, lambda ns=names_found: self.detect_lbl.config(text="\n".join(ns)))
                else:
                    self.after(0, lambda: self.detect_lbl.config(text="Scanning…"))

            elif self.mode == "register":
                locs = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for (top, right, bottom, left) in locs:
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 229, 255), 2)

            # 4. Final Display Resize
            h, w = display.shape[:2]
            scale = min(640/w, 480/h)
            nw, nh = int(w*scale), int(h*scale)
            display = cv2.resize(display, (nw, nh))
            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.after(0, self._update_cam_label, imgtk)

    def _update_cam_label(self, imgtk):
        if hasattr(self, "cam_label") and self.cam_label.winfo_exists():
            self.cam_label.configure(image=imgtk, text="")
            self.cam_label.image = imgtk

    # ─── Reusable button ──────────────────────────────────────────────────────

    @staticmethod
    def _action_btn(parent, label, cmd, color):
        btn = tk.Button(
            parent, text=label, command=cmd,
            bg=color, fg="#000" if color in (ACCENT, SUCCESS, "#F59E0B") else TEXT,
            activebackground=color, relief="flat",
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=12, pady=8, bd=0
        )
        return btn

    # ─── Close hook ───────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_camera()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_attendance_file()
    app = AttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

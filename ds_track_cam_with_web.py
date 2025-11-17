# ds_track_cam_min.py — DeepStream 6.3 (JetPack 5.1.3)
# Defaults: CSI camera @ 640x480@30, mux 640x640, NvDCF tracker
# Usage (only two params kept):
#   python3 ds_track_cam_min.py --pgie pgie_config_yolo.txt --tracker tracker_nvdcf_config.yml
# (You can point --tracker to any file; .yml/.txt both fine.)

import sys, argparse, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import os
import time
from collections import defaultdict, deque
import signal  # <<< added
import Jetson.GPIO as GPIO

from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import json


#GPIO setup

GPIO.setmode(GPIO.BOARD)
COW_GPIO_PIN=29
SHEEP_GPIO_PIN=31
DOG_GPIO_PIN=33
EN_GPIO_PIN=32

for p in (COW_GPIO_PIN, SHEEP_GPIO_PIN, DOG_GPIO_PIN):
    GPIO.setup(p, GPIO.OUT, initial=GPIO.HIGH)
    
GPIO.setup(EN_GPIO_PIN, GPIO.IN)
GPIO.add_event_detect(EN_GPIO_PIN, GPIO.BOTH)

def GPIO_EN_cb(channel):
    level = GPIO.input(EN_GPIO_PIN)
    print(f"EN edge: level={level}")
    if pgie!=None:
        if level:
            print("Start nvinfer")
            pgie.set_property("interval",0)
        else:
            print("Pause nvinfer")
            pgie.set_property("interval",2147483647)
    else:
        print("PGIE has not been initialzed")



GPIO.add_event_callback(EN_GPIO_PIN,GPIO_EN_cb)

pgie=None





_fps_ts = defaultdict(lambda: deque(maxlen=60))

# ---------- helpers ----------

def link_many(*elems):
    for a, b in zip(elems, elems[1:]):
        if not a.link(b): raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
    return True

# ---------- probes ----------
def osd_probe(pad, info, udata):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- FPS calc (per source) ---
        sid = getattr(frame_meta, "source_id", getattr(frame_meta, "pad_index", 0))
        now = time.monotonic()
        ts = _fps_ts[sid]
        ts.append(now)
        fps = 0.0
        if len(ts) >= 2:
            dt = ts[-1] - ts[0]
            if dt > 0:
                fps = (len(ts) - 1) / dt

        #infer status

        interval = pgie.get_property("interval") if pgie else 0
        infer_on = interval == 0

        # --- Text overlay ---
        disp = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        disp.num_labels = 1
        txt = disp.text_params[0]
        txt.display_text = f"Frame {frame_meta.frame_num} | Objects: {frame_meta.num_obj_meta} | FPS: {fps:.1f} | Infer: {'ON' if infer_on else 'OFF'}"
        txt.x_offset = 0
        txt.y_offset = 0
        txt.font_params.font_name = "Sans"
        txt.font_params.font_size = 18
        txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.35)

        pyds.nvds_add_display_meta_to_frame(frame_meta, disp)

        # Update global metrics
        metrics["fps"] = fps
        metrics["frame"] = frame_meta.frame_num

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def largest_box_probe(pad, info, udata):

    KEEP_LABELS = { "cow",  "sheep", "horse"}

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        #  find largest kept object
        largest_area = 0.0
        largest_obj = None
        largest_label = None
        objs = []  # keep references so we can safely remove later

        l_obj = frame_meta.obj_meta_list


        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            objs.append(obj_meta)

            # decode label
            label = obj_meta.obj_label
            if label in KEEP_LABELS:
                r = obj_meta.rect_params
                area = float(r.width) * float(r.height)
                if area > largest_area:
                    largest_area = area
                    largest_obj = obj_meta
                    largest_label = label

            l_obj = l_obj.next

        # remove everything except the largest
        if largest_obj is not None:
            removed = 0
            for om in objs:
                if om is largest_obj:
                    continue
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, om)
                removed += 1
            frame_meta.num_obj_meta = 1
            # optional debug:
           # print(f"Largest kept: {largest_label} area={largest_area:.0f} (removed {removed})")
            #set gpio
            if largest_label=="cow":
                GPIO.output(COW_GPIO_PIN,GPIO.LOW)
                GPIO.output(SHEEP_GPIO_PIN,GPIO.HIGH)
                GPIO.output(DOG_GPIO_PIN,GPIO.HIGH)
                print("COW GPIO")
            elif largest_label=="sheep":
                GPIO.output(SHEEP_GPIO_PIN,GPIO.LOW)
                GPIO.output(COW_GPIO_PIN,GPIO.HIGH)
                GPIO.output(DOG_GPIO_PIN,GPIO.HIGH)
                print("sheep GPIO")
            elif largest_label=="horse":
                GPIO.output(DOG_GPIO_PIN,GPIO.LOW)
                GPIO.output(SHEEP_GPIO_PIN,GPIO.HIGH)
                GPIO.output(COW_GPIO_PIN,GPIO.HIGH)
                print("Horse GPIO")
            else:
                for p in (COW_GPIO_PIN, SHEEP_GPIO_PIN, DOG_GPIO_PIN):
                    GPIO.output(p, GPIO.HIGH)
        else:
            # no kept labels, remove every boxes
            for om in objs:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, om)
            frame_meta.num_obj_meta = 0
            for p in (COW_GPIO_PIN, SHEEP_GPIO_PIN, DOG_GPIO_PIN):
                    GPIO.output(p, GPIO.HIGH)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK

# Metrics for dashboard
metrics = {
    "total": 0,
    "cows": 0,
    "dogs": 0, 
    "sheep": 0,
    "fps": 0,
    "frame": 0,
    "largest_animal": "none",
    "alerts": []
}

from threading import Lock
METRICS_LOCK = Lock()
MAX_ALERTS = 20

def push_alert(msg, now=None):
    if now is None: now = time.time()
    metrics["alerts"].append({"t": now, "msg": msg})
    if len(metrics["alerts"]) > MAX_ALERTS:
        metrics["alerts"] = metrics["alerts"][-MAX_ALERTS:]

# for web server probe
def analytics_probe(pad, info, udata):
    try:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        total = cows = dogs = sheep = 0
        largest_area = 0.0
        largest_label = "none"

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                label = obj_meta.obj_label
                if label:
                    total += 1
                    if label == "cow":   cows += 1
                    elif label == "pig": dogs += 1
                    elif label == "sheep": sheep += 1

                    if label in ("cow","pig","sheep","horse"):
                        r = obj_meta.rect_params
                        area = float(r.width) * float(r.height)
                        if area > largest_area:
                            largest_area = area
                            largest_label = label

                l_obj = l_obj.next

            l_frame = l_frame.next

        # collect alerts WITHOUT lock
        alerts_to_add = []
        cur_fps = metrics.get("fps", 0)  # read without lock is fine for a float
        if cur_fps < 5.0:
            alerts_to_add.append(f"Low FPS: {cur_fps:.1f}")
        if (cows + dogs + sheep) > 10:
            alerts_to_add.append(f"High density: {cows+dogs+sheep} animals")

        # single critical section
        with METRICS_LOCK:
            metrics["total"] = total
            metrics["cows"] = cows
            metrics["dogs"] = dogs
            metrics["sheep"] = sheep
            metrics["largest_animal"] = largest_label
            for a in alerts_to_add:
                push_alert(a, now=time.time())

        return Gst.PadProbeReturn.OK

    except Exception as e:
        # never let exceptions kill the streaming thread
        print(f"[analytics_probe] EXCEPTION: {e}", file=sys.stderr)
        return Gst.PadProbeReturn.OK




# run web server
def run_webserver():
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet logs
            pass

        def _send_json(self, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/metrics", "/metrics.json"):
                with METRICS_LOCK:
                    self._send_json(metrics)
                return
            return super().do_GET()

    httpd = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Metrics at:  http://localhost:8000/metrics.json")
    httpd.serve_forever()

# ---------- main ----------
def main():

    # start web server
    web_thread = threading.Thread(target=run_webserver, daemon=True)
    web_thread.start()

    Gst.init(None)
    pipeline = Gst.Pipeline.new("ds-pipeline")

    # Source (internalized): CSI 640x480@30, NV12/NVMM
    src = Gst.ElementFactory.make("nvarguscamerasrc", "csi-src")
    caps1 = Gst.ElementFactory.make("capsfilter", "csi-caps1")
    caps1.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=NV12,width=1280,height=720"
    ))
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "csi-nvconv")
    # caps2 = make("capsfilter", "csi-caps2")
    # caps2.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
    q_src = Gst.ElementFactory.make("queue", "q-src"); q_src.set_property("leaky", 1); q_src.set_property("max-size-buffers", 4)

    for e in (src, caps1, nvconv1,  q_src):
        pipeline.add(e)
    link_many(src, caps1, nvconv1,  q_src)

    # Streammux → PGIE → NvDCF tracker → nvvideoconvert → nvdsosd → display
    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    mux.set_property("batch-size", 1)
    mux.set_property("live-source", 1)
    mux.set_property("width", 640)   # model input w
    mux.set_property("height", 640)  # model input h
    mux.set_property("batched-push-timeout", 33000)
    pipeline.add(mux)

    sinkpad = mux.get_request_pad("sink_0")
    if not sinkpad: raise RuntimeError("Unable to get sink_0 from nvstreammux")
    srcpad = q_src.get_static_pad("src")
    if not srcpad: raise RuntimeError("Unable to get src pad from source branch")
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link source->mux")

    # --- Downstream: mux → pgie → tracker(NvDCF) → RGBA(NVMM) → nvdsosd(GPU) → RGBA(SYS) → ximagesink ---

    # PGIE
    global pgie
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", "/home/cmpe6012/prg/pgie_config_yolo.txt")

    # NvDCF tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("display-tracking-id", 1)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-6.3/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file","/opt/nvidia/deepstream/deepstream-6.3/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    tracker.set_property("compute-hw",1)

    # Convert to RGBA on NVMM for OSD (GPU mode expects NVMM buffers)
    nv_preosd = Gst.ElementFactory.make("nvvideoconvert", "nv_preosd")
    caps_rgba_nvmm = Gst.ElementFactory.make("capsfilter", "caps_rgba_nvmm")
    caps_rgba_nvmm.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
    )

    # OSD on GPU
    nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
    nvosd.set_property("process-mode", 1)     # 1 = GPU

    # NVMM → SystemMemory for desktop sink
    nv_postosd = Gst.ElementFactory.make("nvvideoconvert", "nv_postosd")
    caps_rgba_sys = Gst.ElementFactory.make("capsfilter", "caps_rgba_sys")
    caps_rgba_sys.set_property(
        "caps", Gst.Caps.from_string("video/x-raw,format=RGBA")
    )
    vconv = Gst.ElementFactory.make("videoconvert", "vconv")

    sink = Gst.ElementFactory.make("ximagesink", "ximg")
    sink.set_property("sync", 0)

    # Add BEFORE linking
    for e in (pgie, tracker, nv_preosd, caps_rgba_nvmm, nvosd, nv_postosd, caps_rgba_sys, vconv, sink):
        pipeline.add(e)

    # Link chain
    link_many(mux, pgie, tracker, nv_preosd, caps_rgba_nvmm, nvosd, nv_postosd, caps_rgba_sys, vconv, sink)


    # OSD text overlay
    osd_sink_pad = nvosd.get_static_pad("sink")   # add meta right before OSD draws
    if not osd_sink_pad:
        raise RuntimeError("Unable to get nvosd sink pad")
    osd_probe_id = osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_probe, None) 
    # Largest-box-only filter
    largest_box_probe_id=osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, largest_box_probe, None)
    #metrics probe
    analytics_probe_id = osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, analytics_probe, None)

    #  mainloop
    loop = GLib.MainLoop()

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping…")
        pipeline.set_state(Gst.State.NULL)
        try:
            osd_sink_pad.remove_probe(osd_probe_id)
            osd_sink_pad.remove_probe(largest_box_probe_id)
            osd_sink_pad.remove_probe(analytics_probe_id)
        except Exception:
            pass
        try:
            mux.release_request_pad(sinkpad)
            sys.exit(0)
        except Exception:
            pass
        sys.exit(0)

if __name__ == "__main__":
    sys.exit(main())

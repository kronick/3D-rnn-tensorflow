import os
from flask import Flask, request, make_response, redirect, url_for, render_template, send_from_directory, jsonify, send_file, g
import json

app = Flask(__name__)

CONFIG = {
  "pointcloud_dir": "."
}

@app.route("/")
def index():
  return render_template("visualize_points.html")

@app.route("/files", methods=["GET"])
def get_filelist():
  files = [f for f in os.listdir(CONFIG["pointcloud_dir"]) if f.endswith(".xyz")]

  files.sort(key=lambda x: os.path.getmtime(os.path.join(CONFIG["pointcloud_dir"], x)), reverse=True)

  return jsonify({"files": files})

@app.route('/files/<filename>/points', methods=["GET"])
def get_points(filename):
  with open(os.path.join(CONFIG["pointcloud_dir"], filename), "r") as f:
    points = []
    for line in f:
      if line.strip() is "":
        continue
      point = [float(p) for p in line.strip().split(" ")]
      points.append(point)

  return jsonify({"points": points})

@app.route('/files/<filename>/states/<int:idx>', methods=["GET"])
def get_states(filename, idx):
  if not filename.endswith(".xyz"):
    return "Must request a .xyz file!", 500
  filename = filename[:-4] + ".states"

  with open(os.path.join(CONFIG["pointcloud_dir"], filename), "r") as f:
    states = []
    minVal = float("inf")
    maxVal = float("-inf")
    for line in f:
      if line.strip() is "":
        continue

      entries = [s for s in line.strip().split(" ") if s is not ""]
      state = float(entries[idx])
      minVal = min(state, minVal)
      maxVal = max(state, maxVal)
      states.append(state)

  return jsonify({"states": states, "min": minVal, "max": maxVal})

@app.route('/files/<filename>/states/', methods=["GET"])
def get_all_states(filename):
  if not filename.endswith(".xyz"):
    return "Must request a .xyz file!", 500
  filename = filename[:-4] + ".states"

  with open(os.path.join(CONFIG["pointcloud_dir"], filename), "r") as f:
    states = []
    # minVals = float("inf")
    # maxVals = float("-inf")
    for line in f:
      if line.strip() is "":
        continue

      state = [float(s) for s in line.strip().split(" ") if s is not ""]
      # minVal = min(state, minVal)
      # maxVal = max(state, maxVal)
      states.append(state)

  return jsonify({"states": states})

app.debug = True
if __name__ == '__main__':
    app.run(host = '0.0.0.0', threaded=True)

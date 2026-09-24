import struct
import pandas as pd
import plotly.graph_objects as graph
from plotly.subplots import make_subplots
import numpy as np


PARAMETERS_RECORD_FORMAT = "<HBI" + "f"*18
FLIGHT_RECORD_FORMAT = "<HBII 4H " + "f"*27 + "5B"


PARAMETERS_RECORD_SIZE   = struct.calcsize(PARAMETERS_RECORD_FORMAT)
FLIGHT_RECORD_SIZE = struct.calcsize(FLIGHT_RECORD_FORMAT)

print("Parameters record length: ",PARAMETERS_RECORD_SIZE)
print("Flight record length", FLIGHT_RECORD_SIZE)

SYNC = 0xA55A
LOG_TYPE_PARAMETERS   = 1
LOG_TYPE_FLIGHT = 2

PARAMETERS_FIELDS = [
    "sync","type","session_id",
    "roll_angle_kp","roll_angle_ki","roll_angle_kd",
    "pitch_angle_kp","pitch_angle_ki","pitch_angle_kd",
    "roll_rate_kp","roll_rate_ki","roll_rate_kd",
    "pitch_rate_kp","pitch_rate_ki","pitch_rate_kd",
    "yaw_rate_kp","yaw_rate_ki","yaw_rate_kd",
    "vertical_velocity_kp","vertical_velocity_ki","vertical_velocity_kd"
]

FLIGHT_FIELDS = [
    "sync","type","session_id","time_us",
    "rc_throttle","rc_roll","rc_pitch","rc_yaw",
    "desired_roll_angle","desired_pitch_angle",
    "desired_roll_rate","desired_pitch_rate","desired_yaw_rate",
    "desired_vertical_velocity",
    "gyro_x","gyro_y","gyro_z",
    "acc_x","acc_y","acc_z",
    "estimated_roll_angle","estimated_pitch_angle",
    "vertical_velocity","altitude",
    "voltage","current",
    "cmd_throttle","cmd_roll","cmd_pitch","cmd_yaw","cmd_hover",
    "m1","m2","m3","m4",
    "is_flying","is_armed","is_radio_failsafe","is_motor_emergency","is_batt_failsafe"
]

# ================== PARSER ==================

def parse_flight_log(filename):

    with open(filename, "rb") as file:
        data = file.read()

    data_len = len(data)

    offset = 0
    last_valid_offset = 0
    last_time = None

    parameters = None
    session_id_from_parameters = None
    logs = []

    while offset + 3 <= data_len:

        # ---- SYNC CHECK ----
        sync = struct.unpack_from("<H", data, offset)[0]
        if sync != SYNC:
            print("Desync at offset", offset)
            break

        log_type = data[offset + 2]

        # ---------- PARAMETERS LOG----------
        if log_type == LOG_TYPE_PARAMETERS:

            if offset + PARAMETERS_RECORD_SIZE > data_len:
                print("PARAMETERS record truncated at", offset)
                break

            raw = struct.unpack_from(PARAMETERS_RECORD_FORMAT, data, offset)

            if raw[0] != SYNC or raw[1] != LOG_TYPE_PARAMETERS:
                print("PARAMETERS record header mismatch at", offset)
                break

            parameters = dict(zip(PARAMETERS_FIELDS, raw))
            session_id_from_parameters = parameters["session_id"]

            print(f"Session ID: {session_id_from_parameters}")

            last_valid_offset = offset + PARAMETERS_RECORD_SIZE
            offset += PARAMETERS_RECORD_SIZE

        # ---------- FLIGHT LOG ----------
        elif log_type == LOG_TYPE_FLIGHT:

            if offset + FLIGHT_RECORD_SIZE > data_len:
                print("Flight record truncated at", offset)
                break

            raw = struct.unpack_from(FLIGHT_RECORD_FORMAT, data, offset)

            if raw[0] != SYNC or raw[1] != LOG_TYPE_FLIGHT:
                print("Flight record header mismatch at", offset)
                break

            record_session_id = raw[2]
            time_us = raw[3]

            # must have PARAMETERS first
            if session_id_from_parameters is None:
                print("FLIGHT record before PARAMETERS record at", offset)
                break

            # ---- SESSION CHECK ----
            if record_session_id != session_id_from_parameters:
                print(
                    f"Session ID mismatch at {offset}: "
                    f"record = {record_session_id}, "
                    f"expected = {session_id_from_parameters}"
                )
                break

            # ---- PREALLOCATED TAIL ----
            if time_us == 0:
                print("Zero timestamp → end of log at", offset)
                break

            # ---- MONOTONIC TIME CHECK ----
            if last_time is not None and time_us < last_time:
                print(
                    "Time reversal at", offset,
                    time_us * 1e-6, "<", last_time * 1e-6
                )
                break

            last_time = time_us
            logs.append(dict(zip(FLIGHT_FIELDS, raw)))

            last_valid_offset = offset + FLIGHT_RECORD_SIZE
            offset += FLIGHT_RECORD_SIZE

        else:
            print("Unknown log type at", offset, "type =", log_type)
            break

    # ================== Validate last record only ==================

    is_last_record_removed = False
    if len(logs) > 0:
        last_row = logs[-1]

        if not record_row_is_valid(last_row):
            print("Last record corrupted -> removing")

            logs.pop()
            is_last_record_removed = True

    # ================== SUMMARY ==================

    if len(logs) > 0:
        print("Last timestamp:", logs[-1]["time_us"] * 1e-6, "s")

    print("\n===== PARSE SUMMARY =====")
    print(f"Last valid offset : {last_valid_offset/1024/1024:.2f} MB")
    print(f"Parsing stopped  : {offset/1024/1024:.2f} MB")
    print(f"File size        : {data_len/1024/1024:.2f} MB")
    print(f"Records parsed   : {len(logs)}")
    print("Last record removed: ", is_last_record_removed)
    print("========================\n")

    return parameters, pd.DataFrame(logs)

# ================== RECORD VALIDATION ==================

def record_row_is_valid(row):
    for v in row.values():
        if isinstance(v, float):
            if not np.isfinite(v) or abs(v) > 1e10:
                return False
    return True

# ------------------------------------------------------

# ================== PLOT ==================

def plot_flight_data(parameters, df_logs):

    if parameters is None:
        print("Parameters is empty")
        return
    
    if df_logs.empty:
        print("Logs is empty")
        return
    

    # ---- convert time to seconds ----
    df_logs["time_s"] = df_logs["time_us"] * 1e-6
    df_logs["dt"] = df_logs["time_s"].diff()
    df_logs["freq"] = 1.0 / df_logs["dt"]

    # -------------------------------
    # PID gains formatted to 3 decimals
    # -------------------------------
    pid_text = (
        "PID gains:<br>"
        f"Roll Angle: Kp={parameters['roll_angle_kp']:.3f}, Ki={parameters['roll_angle_ki']:.3f}, Kd={parameters['roll_angle_kd']:.3f}<br>"
        f"Pitch Angle: Kp={parameters['pitch_angle_kp']:.3f}, Ki={parameters['pitch_angle_ki']:.3f}, Kd={parameters['pitch_angle_kd']:.3f}<br>"
        f"Roll Rate: Kp={parameters['roll_rate_kp']:.3f}, Ki={parameters['roll_rate_ki']:.3f}, Kd={parameters['roll_rate_kd']:.3f}<br>"
        f"Pitch Rate: Kp={parameters['pitch_rate_kp']:.3f}, Ki={parameters['pitch_rate_ki']:.3f}, Kd={parameters['pitch_rate_kd']:.3f}<br>"
        f"Yaw Rate: Kp={parameters['yaw_rate_kp']:.3f}, Ki={parameters['yaw_rate_ki']:.3f}, Kd={parameters['yaw_rate_kd']:.3f}<br>"
        f"Vertical Vel: Kp={parameters['vertical_velocity_kp']:.3f}, Ki={parameters['vertical_velocity_ki']:.3f}, Kd={parameters['vertical_velocity_kd']:.3f}"
    )

    # ======================================================
    # PLOT ALL LOGS
    # ======================================================
    plot_logs = graph.Figure()

    plot_fields = [
        "rc_throttle","rc_roll","rc_pitch","rc_yaw",
        "desired_roll_angle","desired_pitch_angle",
        "desired_roll_rate","desired_pitch_rate","desired_yaw_rate",
        "desired_vertical_velocity",
        "gyro_x","gyro_y","gyro_z",
        "acc_x","acc_y","acc_z",
        "estimated_roll_angle","estimated_pitch_angle",
        "vertical_velocity","altitude",
        "voltage","current",
        "cmd_throttle","cmd_roll","cmd_pitch","cmd_yaw","cmd_hover",
        "m1","m2","m3","m4",
        "is_flying","is_armed","is_radio_failsafe","is_motor_emergency","is_batt_failsafe"
    ]

    for field in plot_fields:
        if field in df_logs.columns:
            plot_logs.add_trace(graph.Scatter(
                x=df_logs["time_s"],
                y=df_logs[field],
                name=field,
                mode="lines",
                visible="legendonly"
            ))

    plot_logs.update_layout(
        title="Flight Logs",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        legend_title="Logs",
        template="plotly_dark",
        paper_bgcolor="#1F2937",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#374151"),
        yaxis=dict(gridcolor="#374151"),
        hovermode="x unified",
        annotations=[dict(
            text=pid_text,
            xref="paper", yref="paper",
            x=0.5, y=1.15,  # above the plot
            showarrow=False,
            align="center",
            font=dict(size=10)
        )]
    )

    config = {
            "toImageButtonOptions": {
            "format": "svg",
            "filename": "flight_logs_svg",
            "scale": 0.5  # change this value to adjust the size of the saved image
        }
    }

    plot_logs.show(config=config)


    # ======================================================
    # PLOT TIME DIFFERENCE & LOOP FREQUENCY
    # ======================================================

    # Create a figure with 2 rows, shared x-axis
    plot_timestamps = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Timestamp Differences", "Loop Frequency")
    )

    # Add dt trace
    plot_timestamps.add_trace(
        graph.Scatter(x=df_logs["time_s"], y=df_logs["dt"], mode="lines", name="dt"),
        row=1, col=1
    )

    # Add frequency trace
    plot_timestamps.add_trace(
        graph.Scatter(x=df_logs["time_s"], y=df_logs["freq"], mode="lines", name="Loop Frequency"),
        row=2, col=1
    )

    # Layout settings
    plot_timestamps.update_layout(
        xaxis2_title="Time (s)",   # bottom subplot x-axis
        yaxis1_title="Δt (s)",     # top subplot y-axis
        yaxis2_title="Loop Frequency (Hz)",  # bottom subplot y-axis
        hovermode="x unified",
        showlegend=True,
        title="Time Differences and Loop Frequency",
        template="plotly_dark",
        paper_bgcolor="#1F2937",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#374151"),
        yaxis=dict(
        range=[0.003, 0.005],
        gridcolor="#374151",
        ),
        yaxis2=dict(
        range=[240, 260],
        gridcolor="#374151",
    ),
    )

    plot_timestamps.show(config=config)

if __name__ == "__main__":
    filename = "flight_log_004.bin"
    parameters, df_logs = parse_flight_log(filename)
    plot_flight_data(parameters, df_logs)

import os
import queue


class State:
    def __init__(self):
        self.directory: str = ""

        self.ca_duration: float = 0.0
        self.exp_duration: float = 0.0
        self.ref_potential: float = 0.0
        self.desired_potential: float = 0.0
        self.expected_resistance: float = 0.0
        self.expected_current: float = 0.0
        self.r_ohm_point = 17
        self.self_correction = True
        self.calculate_ct = True

        self.debug=False
        self.first_file_arrived = False
        self.first_file_processed = False
        self.global_time = 0.0

        self.last_ir_result = None
        self.last_curr_result = None
        self.imp_times = []

        self.first_semi_ir_results = {
            "Zir": [],
            "t": []
        }
        self.second_semi_ir_results = {
            "Zct": [],
            "t": []
        }

        self.ca_results = {
            "Iavg": [],
            "t": [],
            "std(I)": []
        }


        self.electrode_value: float = self.ref_potential + self.desired_potential
        self.expected_electrode_value = self.electrode_value + (self.expected_current * self.expected_resistance / 1e3)


        # self.parameter_file_dir = r"\Users\mihah\OneDrive\Kemijski inštitut\programiranje\blaz_impedanca\psmethods"
        self.parameter_file_dir = r"C:\peak454-64bitWin10\Cu foil\operanso impedance"
        self.parameter_file_type = ".psmethod"

        self.parameter_file_names = [
            "CA",
            "CA calibration", # 2 s
            "impedance at potential",
            "Trial impedance",
            "CA cal" # 4 s
        ]

        self.parameter_files = [os.path.join(self.parameter_file_dir, x + self.parameter_file_type) for x in self.parameter_file_names]

    def recalculate_electrode_value(self):
        self.electrode_value = self.ref_potential + self.desired_potential
        self.expected_electrode_value = self.electrode_value + (self.expected_current * self.expected_resistance / 1e3)

STATE = State()

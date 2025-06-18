import queue
import threading
import logging

from watchdog.observers import Observer

from file_handler import NewFileHandler
from workdelegator import WorkDelegator
from state import STATE
import sys
import pandas as pd
from update_parameter_file import UpdateParameterFile
import matplotlib.pyplot as plt
import os


class Plotter:
    def __init__(self, rows: int = 3, cols: int = 1, fig_size: tuple = (10, 8)):
        self.rows = rows if STATE.calculate_ct else 2
        self.cols = cols
        print(self.rows)
        self.fig_size = fig_size
        plt.ion()
        self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=fig_size)
        plt.tight_layout()
        plt.draw()
        self.xs = None
        self.ys = None
        self.weird_data = False

    def plot_redraw(self, data):

        self.xs, self.ys = data
        self.clear_plots()

        self.plot_impedance()
        if STATE.calculate_ct:
            self.plot_second_impedance()
        self.plot_average_current()
        self.update_state()

        plt.tight_layout()
        plt.draw()
        
        plt.savefig(os.path.join(STATE.directory, "Figure.png"),dpi=300)
        self.save_data()
        self.weird_data = False


    def clear_plots(self):
        for ax in self.axs:
            ax.clear()

    def save_data(self):
        data = {
            "t (s)": self.xs["imp_x"],
            "Rohm (Ohm)": self.ys["imp_y"]
        }
        impedance = pd.DataFrame(data)
        if STATE.calculate_ct:
            data_2 = {
                "t (s)": self.xs["imp_second_x"],
                "Rct (Ohm)": self.ys["imp_second_y"]
            }
            ct = pd.DataFrame(data_2)

        data_3 = {
            "Iavg (mA)": self.ys["I_avg_y"],
            "t (s)": self.xs["I_avg_x"],
            "std": self.ys["I_std_y"]
        }
        ca = pd.DataFrame(data_3)

        with pd.ExcelWriter(os.path.join(STATE.directory, "Data.xlsx")) as writer:
            impedance.to_excel(writer,sheet_name='Data',startrow=0 , startcol=0)
            if STATE.calculate_ct:
                ct.to_excel(writer,sheet_name='Data',startrow=0, startcol=5)
            ca.to_excel(writer,sheet_name='Data',startrow=0, startcol=10)

    def plot_impedance(self):
        self.axs[0].plot(
            self.xs["imp_x"],
            self.ys["imp_y"],
            color="blue",
            label="Impedance",
            marker="o",
            linestyle="-",
        )
        self.axs[0].set_title("ROhm Value vs Time")
        self.axs[0].set_xlabel("Time (s)")
        self.axs[0].set_ylabel("Impedance Value")
        self.axs[0].legend()

    def plot_second_impedance(self):
        self.axs[1].plot(
            self.xs["imp_second_x"],
            self.ys["imp_second_y"],
            color="purple",
            label="Impedance",
            marker="o",
            linestyle="-",
        )
        self.axs[1].set_title("RCT Value vs Time")
        self.axs[1].set_xlabel("Time (s)")
        self.axs[1].set_ylabel("Impedance Value")
        self.axs[1].legend()


    def plot_average_current(self):

        if len(self.ys["I_avg_y"]) > 1 and abs(self.ys["I_avg_y"][-1] - self.ys["I_avg_y"][-2]) > 5 :
            del self.ys["I_avg_y"]
            del self.xs["I_avg_x"]
            del self.ys["I_std_y"]
            self.weird_data = True
        else:
        
            self.axs[self.rows-1].errorbar(
                self.xs["I_avg_x"],
                self.ys["I_avg_y"],
                yerr=self.ys["I_std_y"],
                fmt="-o",
                color="green",
                capsize=3,
                ecolor='green',
                label="Average Current with Std Dev",
            )
            self.axs[self.rows-1].set_title("Average Current vs Time with Error Bars")
            self.axs[self.rows-1].set_xlabel("Time (s)")
            self.axs[self.rows-1].set_ylabel("Average Current (mA)")
            self.axs[self.rows-1].legend()

    def update_state(self):
        STATE.first_semi_ir_results = {
            "t": self.xs["imp_x"],
            "Zir": self.ys["imp_y"]
        }
        if len(self.ys["imp_y"]) > 0 and not self.weird_data:
            STATE.last_ir_result = self.ys["imp_y"][-1]
        if STATE.calculate_ct:
            STATE.second_semi_ir_results = {
                "t": self.xs["imp_second_x"],
                "Zct": self.ys["imp_second_y"]
            }

        STATE.ca_results = {
            "t": self.xs["I_avg_x"],
            "Iavg": self.ys["I_avg_y"],
            "std(I)": self.ys["I_std_y"]
        }
        if len(self.ys["I_avg_y"]) > 0:
            STATE.last_curr_result = self.ys["I_avg_y"][-1]

        


class Application:
    def __init__(self):
        self.ca_pf = UpdateParameterFile(STATE.parameter_files[0])
        self.ca_calibration_pf = UpdateParameterFile(STATE.parameter_files[1])
        self.impedance_at_potential = UpdateParameterFile(STATE.parameter_files[2])
        self.impedance_trial = UpdateParameterFile(STATE.parameter_files[3])
        # TODO: SPROBI ČE DELUJE ZA 2 SEKUNDI IN NI TREBA PRVO MET 4 IN POL 2
        self.file_queue = queue.Queue()
        self.data_to_plot_queue = queue.Queue()

    def input_initial_parameters(self):
        is_adjust = input("Did you adjust the parameters of the electrochemical metods? (y/n) ")
        if is_adjust != "y" and is_adjust != "Y" and is_adjust != "yes" and is_adjust != "Yes":
            sys.exit("You need to adjust the parameters of the electrochemical metods!")

        self_correction = input("Self correction for new Rohm? (y/n) ")
        if self_correction != "y" and self_correction != "Y" and self_correction != "yes" and self_correction != "Yes":
            STATE.self_correction = False
            STATE.expected_resistance = float(input("Enter the expected ROhm (in Ohms): "))

        ct_calc = input("Should the program calculate Rct (from Rohm value to max)? (y/n) ")
        if ct_calc != "y" and ct_calc != "Y" and ct_calc != "yes" and ct_calc != "Yes":
            STATE.calculate_ct = False
            
        
        try:

            STATE.directory = input("Please provide the path to the directory to monitor for new files: ")

            STATE.ca_duration = float(input("Enter the duration for the first CA (in seconds): "))
            STATE.exp_duration = float(input("Enter the duration for CA plus imp (in seconds): "))

            STATE.ref_potential = float(input("Enter the potential of the reference electrode (in volts): "))
            STATE.desired_potential = float(input("Enter the desired potential (in volts): "))
            if STATE.self_correction:
                STATE.expected_resistance = float(input("Enter the expected ROhm (in Ohms) or write number zero (0): "))
            try:
                points = int(input("How many points for Rohm calculation (default = 17): "))
            except ValueError:
                points = 17
            finally:
                STATE.r_ohm_point = points
            
            STATE.expected_current = float(
                input("Enter the expected current at this potential (in mA) or write number zero (0): "))
            STATE.last_ir_result = STATE.expected_resistance
            STATE.last_curr_result = STATE.expected_current
            STATE.recalculate_electrode_value()
        except ValueError:
            sys.exit("Invalid input. Please try again.")
        finally:
            self.initialize_parameter_files()

    def initialize_parameter_files(self):
        self.ca_pf.set_parameter("IR_DROP", STATE.last_ir_result)
        self.ca_pf.set_parameter("T_RUN", STATE.ca_duration)
        self.ca_pf.set_parameter("E_VALUE", STATE.electrode_value)
        # self.ca_pf.set_parameter("E_BEGIN", STATE.electrode_value)
        self.ca_pf.set_parameter("E_STBY", STATE.expected_electrode_value)
        self.ca_pf.update()

        self.ca_calibration_pf.set_parameter("IR_DROP", None)
        self.ca_calibration_pf.set_parameter("T_RUN", 2.0)
        self.ca_calibration_pf.set_parameter("E_VALUE", STATE.expected_electrode_value)
        # self.ca_calibration_pf.set_parameter("E_BEGIN", STATE.expected_electrode_value)
        self.ca_calibration_pf.set_parameter("E_STBY", STATE.expected_electrode_value)
        self.ca_calibration_pf.update()

        self.impedance_at_potential.set_parameter("IR_DROP", None)
        self.impedance_at_potential.set_parameter("T_RUN", None)
        self.impedance_at_potential.set_parameter("E_VALUE", STATE.expected_electrode_value)
        # self.impedance_at_potential.set_parameter("E_BEGIN", STATE.expected_electrode_value)
        self.impedance_at_potential.set_parameter("E_STBY", STATE.expected_electrode_value)
        self.impedance_at_potential.update()

        self.impedance_trial.set_parameter("IR_DROP", None)
        self.impedance_trial.set_parameter("T_RUN", None)
        self.impedance_trial.set_parameter("E_COND", STATE.expected_electrode_value)
        self.impedance_trial.set_parameter("E_VALUE", STATE.expected_electrode_value)
        # self.impedance_trial.set_parameter("E_BEGIN", STATE.expected_electrode_value)
        self.impedance_trial.set_parameter("E_STBY", STATE.expected_electrode_value)
        self.impedance_trial.update()

    def run(self):
        self.input_initial_parameters()
        logging.warning(f"Monitoring directory: {STATE.directory}")

        event_handler = NewFileHandler(self.file_queue)
        work_delegator = WorkDelegator()
        plotter = Plotter()

        observer = Observer()
        observer.schedule(event_handler, STATE.directory, recursive=False)
        observer.start()

        proccess_thread = threading.Thread(target=work_delegator.proccessing_worker,
                                           args=(self.file_queue, self.data_to_plot_queue,), daemon=True)
        proccess_thread.start()
        try:
            while True:
                try:
                    data = self.data_to_plot_queue.get(timeout=0.1)
                    if data is None:
                        break

                    plotter.plot_redraw(data)


                    diff_imp_time = 0

                    if len(STATE.imp_times) >= 1:
                        diff_imp_time = (len(STATE.imp_times)-1)*STATE.exp_duration - STATE.imp_times[-1]

                    if STATE.self_correction:
                        e_compensated = STATE.electrode_value + (STATE.last_ir_result * STATE.last_curr_result / 1e3)
                    else:
                        e_compensated = STATE.electrode_value + (STATE.expected_resistance * STATE.last_curr_result / 1e3)
                    self.ca_pf.set_parameter("T_RUN", STATE.ca_duration if diff_imp_time == 0 else STATE.ca_duration + diff_imp_time)
                    self.ca_pf.set_parameter("E_VALUE", STATE.electrode_value)
                    if STATE.self_correction:
                        self.ca_pf.set_parameter("IR_DROP", STATE.last_ir_result)
                    else:
                        self.ca_pf.set_parameter("IR_DROP", STATE.expected_resistance)
                    self.ca_pf.set_parameter("E_STBY", e_compensated)
                    self.ca_pf.update()

                    self.ca_calibration_pf.set_parameter("IR_DROP", None)
                    self.ca_calibration_pf.set_parameter("T_RUN", 2.0)
                    self.ca_calibration_pf.set_parameter("E_VALUE", e_compensated)
                    self.ca_calibration_pf.set_parameter("E_STBY", e_compensated)
                    self.ca_calibration_pf.update()

                    self.impedance_at_potential.set_parameter("IR_DROP", None)
                    self.impedance_at_potential.set_parameter("T_RUN", None)
                    self.impedance_at_potential.set_parameter("E_VALUE", e_compensated)
                    self.impedance_at_potential.set_parameter("E_STBY", e_compensated)
                    self.impedance_at_potential.update()



                    
                except queue.Empty:
                    pass

                plt.pause(0.5)

                # threading.Event().wait(1)
        except KeyboardInterrupt:
            observer.stop()
            self.file_queue.put(None)
            self.data_to_plot_queue.put(None)
        observer.join()
        logging.warning("Directory worker received exit signal. Shutting down.")
        logging.info("Received stop signal, exiting plot loop.")
        proccess_thread.join()

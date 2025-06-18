import queue
import time

import logging
import ntpath

from data_extractor import DataExtractor
import numpy as np

from state import STATE
import time
import traceback


class WorkDelegator:
    def __init__(self):
        self.xs = {
            "I_avg_x": [],
            "I_std_x": [],
            "imp_x": [],
            "imp_second_x": []
        }
        self.ys = {
            "I_avg_y": [],
            "I_std_y": [],
            "imp_y": [],
            "imp_second_y": []
        }
        self.is_first_imp = False

    def process_file(self, filepath: str):
        logging.info(f"Processing file: {filepath}")
        filename = ntpath.basename(filepath)
        if not STATE.first_file_arrived:
            STATE.first_file_arrived = True
            STATE.global_time = time.time()
        if "CA" in filename and ntpath.isfile(filepath):
            I, t, E = DataExtractor.extract_data_ca(filepath)

            running_sum = 0
            values = []
            
            for i in range(len(I)):
                if I[i] > -28:
                    running_sum += I[i]
                    values.append(I[i])
            
            if len(values) > 5:
                values = np.array(values)
                I_avg = running_sum/len(values)
                I_std = np.std(values)

                self.xs["I_avg_x"].append(time.time() - STATE.global_time)
                self.ys["I_avg_y"].append(I_avg)
                self.xs["I_std_x"].append(time.time() - STATE.global_time)
                self.ys["I_std_y"].append(I_std)
                logging.info(f"Average current: {I_avg:.2f} mA")
                logging.info(f"Standard deviation of current: {I_std:.2f} mA")
                STATE.ca_results["Iavg"].append(I_avg)
                STATE.ca_results["std(I)"].append(I_std)
                STATE.ca_results["t"].append(time.time() - STATE.global_time)
            else:
                t_now = time.time() - STATE.global_time
                self.xs.setdefault("I_avg_x", [])
                self.xs.setdefault("I_std_x", [])
                self.ys.setdefault("I_avg_y", [])
                self.ys.setdefault("I_std_y", [])

                if STATE.ca_results["Iavg"]:
                    last_iavg = STATE.ca_results["Iavg"][-1]
                    last_std = STATE.ca_results["std(I)"][-1]
                    logging.warning("CA invalid, reusing last valid CA result.")
                else:
                    last_iavg = STATE.expected_current
                    last_std = 0.0
                    logging.warning("CA invalid, using expected current as fallback.")

                self.xs["I_avg_x"].append(t_now)
                self.ys["I_avg_y"].append(last_iavg)
                self.xs["I_std_x"].append(t_now)
                self.ys["I_std_y"].append(last_std)

                logging.info(f"Average current: {last_iavg:.2f} mA")
                logging.info(f"Standard deviation of current: {last_std:.2f} mA")
                
        if "imp" in filename and ntpath.isfile(filepath):
            ir_time = time.time() - STATE.global_time
            STATE.imp_times.append(ir_time)

            result_ir = DataExtractor.extract_data_imp_new(filepath)
            if not isinstance(result_ir, (int, float)):
                logging.warning("Impedance IR result invalid. Using last valid result.")
                result_ir = STATE.last_ir_result

            if result_ir < 1 or result_ir > 50:
                logging.warning("Impedance IR result below 1 or above 50! Using last valid result.")
                result_ir = STATE.last_ir_result

            if STATE.calculate_ct:
                result_second_semi = DataExtractor.extract_data_imp_second_semi(filepath, result_ir)

            self.xs["imp_x"].append(ir_time)
            self.ys["imp_y"].append(result_ir)
            logging.info(f"Ir drop: {result_ir:.2f} Ohm")
            
            
            
            if self.is_first_imp and STATE.calculate_ct:
                
                if isinstance(result_second_semi,float) or isinstance(result_second_semi,int): 
                    
                    self.xs["imp_second_x"].append(ir_time)
                    ir_second_semi = (result_second_semi - result_ir) * 2
                    self.ys["imp_second_y"].append(ir_second_semi)
                    logging.info(f"CT resistance: {ir_second_semi:.2f} Ohm")
            else:
                self.is_first_imp = True

        return [self.xs, self.ys]


    def proccessing_worker(self, file_to_process: queue.Queue, data_to_plot: queue.Queue):
        while True:
            try:
                filepath = file_to_process.get()
                if filepath is None:
                    logging.warning("Plotting worker received exit signal. Shutting down.")
                    file_to_process.task_done()
                    break
                filename = ntpath.basename(filepath)
                if "imp" in filename or "CA" in filename:
                    data = self.process_file(filepath)
                    data_to_plot.put(data)
                    file_to_process.task_done()
            except Exception as e:
                
                logging.error(f"Error processing file {filepath}: {traceback.format_exc()}")
                file_to_process.task_done()


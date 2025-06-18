import json
import unicodedata
import logging
import numpy as np
import pandas as pd
import statistics

class DataExtractor:

    @staticmethod
    def remove_control_chars(s):
        return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')

    @staticmethod
    def convert_potential_unit(from_unit, to_unit):
        conversion_factors = {
            ('MilliVolt', 'Volt'): 0.001,
            ('Volt', 'MilliVolt'): 1000,
            ('Volt', 'MicroVolt'): 1_000_000,
            ('MicroVolt', 'Volt'): 0.000001,
            ('MilliVolt', 'MicroVolt'): 1000,
            ('MicroVolt', 'MilliVolt'): 0.001,
            ('Volt', 'Volt'): 1,
            ('MilliVolt', 'MilliVolt'): 1,
            ('MicroVolt', 'MicroVolt'): 1,
        }
        factor = conversion_factors.get((from_unit, to_unit))
        if factor is None:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")
        return factor

    @staticmethod
    def convert_current_unit(from_unit, to_unit):
        conversion_factors = {
            ('MilliAmpere', 'Ampere'): 0.001,
            ('Ampere', 'MilliAmpere'): 1000,
            ('Ampere', 'MicroAmpere'): 1_000_000,
            ('MicroAmpere', 'Ampere'): 0.000001,
            ('MilliAmpere', 'MicroAmpere'): 1000,
            ('MicroAmpere', 'MilliAmpere'): 0.001,
            ('Ampere', 'Ampere'): 1,
            ('MilliAmpere', 'MilliAmpere'): 1,
            ('MicroAmpere', 'MicroAmpere'): 1,
        }
        factor = conversion_factors.get((from_unit, to_unit))
        if factor is None:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")
        return factor

    @staticmethod
    def get_json_from_psfile(file_path: str):
        obj = None
        with open(file_path, 'r', encoding="utf-16") as f:
            for line_number, line in enumerate(f, start=1):
                clean_line = DataExtractor.remove_control_chars(line.strip())
                if not clean_line:  # Skip empty lines after cleanup
                    continue
                try:
                    obj = json.loads(clean_line)
                except json.JSONDecodeError as e:
                    print(f'Error parsing JSON object at line {line_number}: {e}')
        return obj

    @staticmethod
    def extract_data_ca(file_path: str):

        I = []
        t = []
        E = []

        obj = DataExtractor.get_json_from_psfile(file_path)
        measurement_data = obj.get('Measurements', [])
        data_set = measurement_data[0].get('DataSet', {}) if measurement_data else {}
        values = data_set.get('Values', []) if data_set else []

        for value in values:
            if value.get('Description') == 'time':
                # The unit should always be seconds
                time_values = value.get('DataValues', [])
                for pair in time_values:
                    t.append(pair['V'])
            if value.get('Description') == 'potential':
                # The unit should always be Volt
                unit = value.get('Unit').get("Type")
                unit = unit.split(".")[-1]
                conversion_factor = DataExtractor.convert_potential_unit(unit, 'Volt')
                potential_values = value.get('DataValues', [])
                for pair in potential_values:
                    E.append(pair['V'] * conversion_factor)

            if value.get('Description') == 'current':
                # The unit should always be Volt
                unit = value.get('Unit').get("Type")
                unit = unit.split(".")[-1]
                conversion_factor = DataExtractor.convert_current_unit(unit, 'MilliAmpere')
                potential_values = value.get('DataValues', [])
                for pair in potential_values:
                    I.append(pair['V'] * conversion_factor)

        logging.info(f"Extracted values from CA file {file_path}")

        I = np.array(I)
        t = np.array(t)
        E = np.array(E)

        return I, t, E

    @staticmethod
    def extract_data_imp(file_path: str):

        # Data extraction for ZRe and ZIm
        zre_values, zim_values = [], []

        obj = DataExtractor.get_json_from_psfile(file_path)
        print(obj)
        measurement_data = obj.get('Measurements', [])
        data_set = measurement_data[0].get('DataSet', {}) if measurement_data else {}
        values = data_set.get('Values', []) if data_set else []
        for value in values:
            # Both values are belived to be in ohms always
            if value.get('Description') == 'ZRe':
                zre_vals = value.get('DataValues', [])
                for pair in zre_vals:
                    zre_values.append(pair['V'])
            if value.get('Description') == 'ZIm':
                zim_vals = value.get('DataValues', [])
                for pair in zim_vals:
                    zim_values.append(pair['V'])
        # Create a DataFrame with columns ZRe and ZIm
        df = pd.DataFrame({'ZRe': zre_values[:17], 'ZIm': zim_values[:17]})

        # Remove rows where either ZRe or ZIm contains a negative value
        df = df[(df['ZRe'] >= 0) & (df['ZIm'] >= 0)]

        if len(df) >= 6:
            x = range(2, 6)
            y = df['ZIm'].iloc[2:6]
            poly_coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(poly_coeffs)

            # Check if the first or first two points deviate from the trendline by more than ±1
            outliers = []
            for i in range(2):  # Only check the first two points
                if abs(df.iloc[i]['ZIm'] - poly(i)) > 1:
                    outliers.append(i)

            # Remove identified outliers
            if outliers:
                df = df.drop(outliers).reset_index(drop=True)

        # Initialize list to store indices of rows to keep
        rows_to_keep = [0]
        a = True
        j = 0
        c = 0
        # Check middle rows and apply conditions
        for i in range(1, len(df) - 1):  # Exclude the first row initially

            current_zre = df.iloc[i]['ZRe']
            next_zre = df.iloc[i + 1]['ZRe']
            if a == False:
                i = i - j
                previous_zre = df.iloc[i - 1]['ZRe']
                if previous_zre - 1 < current_zre < next_zre + 1:
                    rows_to_keep.append(i + j)
                    a = True
                else:
                    a = False
                    j = j + 1

            # Keep the row if there is a lower previous ZRe or a higher next ZRe
            else:
                previous_zre = df.iloc[i - 1]['ZRe']
                if previous_zre - 1 < current_zre < next_zre + 1:
                    rows_to_keep.append(i)
                    a = True
                else:
                    a = False
                    j = j + 1
            if a == False:
                c = c + 1
                if c > 3:
                    break
            else:
                c = 0

        # Add the first row and the last row to rows_to_keep
        rows_to_keep = rows_to_keep + [len(df) - 1]
        # Check if the last row in rows_to_keep has a ZRe value lower than the one before it
        if df.iloc[rows_to_keep[-1]]['ZRe'] < df.iloc[rows_to_keep[-2]]['ZRe']:
            rows_to_keep.pop()  # Remove the last row if it fails the check

        # Further filtering based on ZIm trendline
        refined_rows = []
        for idx in rows_to_keep:
            if idx == 0:
                # First row: use the next three points
                fit_indices = [1, 2, 3]
            elif idx == 1:
                # Second row: use itself and the two points after
                fit_indices = [0, 2, 3]
            elif idx == len(df) - 1 or idx == len(df) - 2:
                # Last or second-to-last row: ignore since we can't fit a full polynomial
                refined_rows.append(idx)
                continue
            else:
                # Middle rows: use two points before and two after
                fit_indices = [idx - 2, idx - 1, idx + 1, idx + 2]

            # Get ZRe and ZIm values for polynomial fitting
            x = df.iloc[fit_indices]['ZRe']
            y = df.iloc[fit_indices]['ZIm']

            # Fit a polynomial of degree 3
            poly_coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(poly_coeffs)

            # Calculate the expected ZIm value at the current ZRe
            expected_zim = poly(df.iloc[idx]['ZRe'])

            # Check if the current ZIm is within ±1 of the trendline
            if abs(df.iloc[idx]['ZIm'] - expected_zim) <= 1:
                refined_rows.append(idx)
            else:
                fit_indices = [idx - 3, idx - 2, idx - 1]
                poly_coeffs = np.polyfit(x, y, 2)
                poly = np.poly1d(poly_coeffs)
                if abs(df.iloc[idx]['ZIm'] - expected_zim) <= 1:
                    refined_rows.append(idx)
                else:
                    fit_indices = [idx + 3, idx + 2, idx + 1]
                    poly_coeffs = np.polyfit(x, y, 2)
                    poly = np.poly1d(poly_coeffs)
                    if abs(df.iloc[idx]['ZIm'] - expected_zim) <= 1:
                        refined_rows.append(idx)

        # Filter the DataFrame based on refined indices
        df = df.iloc[refined_rows].reset_index(drop=True)

        if len(df) < 4:
            return "Not enough data"

        min_index = df['ZIm'].idxmin()
        y_min = df['ZIm'].iloc[min_index]
        x_at_min_y = df['ZRe'].iloc[min_index]

        a = True
        b = False
        if min_index != 0:
            if df['ZIm'].iloc[-1] == y_min:
                x_min = df['ZRe'].iloc[min_index]
                a = False

            if df['ZIm'].iloc[-1] != y_min and df['ZIm'].iloc[-1] - y_min <= 0.5:
                x_min = (df['ZRe'].iloc[min_index] + df['ZRe'].iloc[-1]) / 2
                a = False

            if a == True:
                x = df['ZRe'].iloc[:]
                y = df['ZIm'].iloc[:]
                poly_coeffs = np.polyfit(x, y, 4)
                poly_4 = np.poly1d(poly_coeffs)
                x_plot_p4 = np.linspace(df['ZRe'].iloc[0], df['ZRe'].iloc[-1], 25)
                y_plot_p4 = poly_4(x_plot_p4)
                min_index_p4 = np.argmin(y_plot_p4)
                x_min_p4 = x_plot_p4[min_index_p4]

                # Select the points: one before, the minimum, and one after
                x_values = df['ZRe'].iloc[min_index - 1: min_index + 2]
                y_values = df['ZIm'].iloc[min_index - 1: min_index + 2]

                # Fit a second-degree polynomial (quadratic) through these three points
                coefficients = np.polyfit(x_values, y_values, 2)
                poly_2 = np.poly1d(coefficients)
                # Get the coefficients of the polynomial
                a, b, c = coefficients  # where coefficients = [a, b, c] from np.polyfit

                # Calculate the x-coordinate of the minimum
                x_min_p2 = -b / (2 * a)
                x_plot_p2 = np.linspace(df['ZRe'].iloc[min_index - 1], df['ZRe'].iloc[min_index + 1], 20)
                y_plot_p2 = poly_2(x_plot_p2)
                b = True
                a = True

        else:
            # Assuming `df` is your DataFrame and has columns 'ZRe' and 'ZIm'
            x = df['ZRe'].iloc[:]
            y = df['ZIm'].iloc[:]

            # Step 1: Fit a 4th order polynomial to the data
            poly_coeffs = np.polyfit(x, y, 4)
            poly_4 = np.poly1d(poly_coeffs)

            # Step 2: Compute the first derivative of the polynomial
            poly_4_deriv = poly_4.deriv()

            # Step 3: Find the roots of the derivative to locate points where the slope is zero
            critical_points = poly_4_deriv.roots
            real_critical_points = [point for point in critical_points if np.isreal(point)]
            real_critical_points = np.real(real_critical_points)  # Extract real parts

            # Filter critical points to only include those within the observed range of x values
            x_min_val, x_max_val = x.min(), x.max()
            real_critical_points = [point for point in real_critical_points if x_min_val <= point <= x_max_val]

            # Step 4: Check for local minima and maxima by evaluating the derivative around each critical point
            local_minima = []
            local_maxima = []

            for point in real_critical_points:
                # Evaluate the derivative slightly to the left and right of the point
                left_of_point = poly_4_deriv(point - 1e-4)
                right_of_point = poly_4_deriv(point + 1e-4)

                # Check if it is a local minimum (derivative changes from negative to positive)
                if left_of_point < 0 and right_of_point > 0:
                    local_minima.append(point)

                # Check if it is a local maximum (derivative changes from positive to negative)
                if left_of_point > 0 and right_of_point < 0:
                    local_maxima.append(point)

            # Step 5: Process based on whether a local minimum was found
            if local_minima and local_maxima:

                # Set x_min_p4 to the first local minimum that meets the criteria
                x_min_p4 = local_minima[0]

                lower_bound = local_minima[0] - 2
                upper_bound = local_minima[0] + 6
                df_in_range = df[(df['ZRe'] >= lower_bound) & (df['ZRe'] <= upper_bound)]
                if not df_in_range.empty:
                    min_index = df_in_range['ZIm'].idxmin()
                    y_min = df_in_range['ZIm'].loc[min_index]
                    x_at_min_y = df_in_range['ZRe'].loc[min_index]

                # Step 7: Fit a second-degree polynomial around x_at_min_y and calculate x_min_p2
                # Ensure min_index has surrounding points to perform the fit
                if min_index > 0 and min_index < len(df) - 1:
                    x_values = df['ZRe'].iloc[min_index - 1: min_index + 2]
                    y_values = df['ZIm'].iloc[min_index - 1: min_index + 2]

                    # Fit a second-degree polynomial (quadratic) through these three points
                    coefficients = np.polyfit(x_values, y_values, 2)
                    poly_2 = np.poly1d(coefficients)

                    # Calculate the x-coordinate of the minimum of the quadratic polynomial
                    a, b, c = coefficients  # where coefficients = [a, b, c] from np.polyfit
                    x_min_p2 = -b / (2 * a)

                    # Optional: generate values for plotting the second-degree polynomial
                    x_plot_p2 = np.linspace(df['ZRe'].iloc[min_index - 1], df['ZRe'].iloc[min_index + 1], 20)
                    y_plot_p2 = poly_2(x_plot_p2)
                    a = True
                    b = True

                else:
                    logging.error("Not enough data points to fit a second-degree polynomial around the minimum ZIm.")


            else:
                # Assuming 'df' is your DataFrame and has columns 'ZRe' and 'ZIm'
                x = df['ZRe'].iloc[:]
                y = df['ZIm'].iloc[:]

                # Step 1: Fit a 3rd-order polynomial to the data
                poly_coeffs = np.polyfit(x, y, 3)
                poly_3 = np.poly1d(poly_coeffs)

                # Step 2: Compute the first derivative of the polynomial
                poly_3_deriv = poly_3.deriv()

                # Step 3: Compute the second derivative of the polynomial (for finding critical points of the first derivative)
                poly_3_deriv2 = poly_3_deriv.deriv()

                # Step 4: Find the roots of the second derivative to locate critical points of the first derivative
                critical_points = poly_3_deriv2.roots
                real_critical_points = [point for point in critical_points if np.isreal(point)]
                real_critical_points = np.real(real_critical_points)  # Extract real parts

                # Step 5: Filter critical points within the observed range of x values
                x_min_val, x_max_val = x.min(), x.max()
                real_critical_points = [point for point in real_critical_points if x_min_val <= point <= x_max_val]

                # Step 6: Identify the minimum of the first derivative by evaluating the third derivative at each critical point
                poly_3_deriv3 = poly_3_deriv2.deriv()  # Third derivative for checking minima
                x_min = None
                min_deriv_value = float('inf')  # Start with a large value

                for point in real_critical_points:
                    # Check if this point is a minimum of the first derivative
                    if poly_3_deriv3(point) > 0:  # Positive third derivative implies a local minimum
                        deriv_value = poly_3_deriv(point)  # Value of the first derivative at this point
                        if deriv_value < min_deriv_value:  # Find the smallest value of the first derivative
                            min_deriv_value = deriv_value
                            x_min = point

                # Debug: Print the found minimum point of the first derivative
                if x_min is not None:
                    logging.info(
                        f"Minimum of the first derivative found at x = {x_min}, with derivative value = {min_deriv_value}")
                else:
                    logging.info("No minimum of the first derivative found within the x range.")

                b = False
                a = False

        if a == False:
            return x_min
        if b == True:

            avg = (x_at_min_y + x_min_p4 + x_min_p2) / 3
            data_R = [x_at_min_y, x_min_p4, x_min_p2]
            if abs(statistics.stdev(data_R) / avg) >= 0.08:
                dev_p2 = abs(x_min_p2 - avg)
                dev_p4 = abs(x_min_p4 - avg)
                dev_min = abs(x_at_min_y - avg)
                dev_avg = [dev_min, dev_p4, dev_p2]

                max_dev_index = dev_avg.index(max(dev_avg))

                # Remove the item in data_R at this index
                data_R = [v for i, v in enumerate(data_R) if i != max_dev_index]
                avg = sum(data_R) / len(data_R)
            return avg

    @staticmethod
    def extract_data_imp_new(file_path: str):

        # Data extraction for ZRe and ZIm
        zre_values, zim_values = [], []

        obj = DataExtractor.get_json_from_psfile(file_path)
        measurement_data = obj.get('Measurements', [])
        data_set = measurement_data[0].get('DataSet', {}) if measurement_data else {}
        values = data_set.get('Values', []) if data_set else []
        for value in values:
            # Both values are belived to be in ohms always
            if value.get('Description') == 'ZRe':
                zre_vals = value.get('DataValues', [])
                for pair in zre_vals:
                    zre_values.append(pair['V'])
            if value.get('Description') == 'ZIm':
                zim_vals = value.get('DataValues', [])
                for pair in zim_vals:
                    zim_values.append(pair['V'])
        # Create a DataFrame with columns ZRe and ZIm
        df = pd.DataFrame({'ZRe': zre_values[:17], 'ZIm': zim_values[:17]})

        # Remove rows where either ZRe or ZIm contains a negative value
        df = df[(df['ZRe'] >= 0) & (df['ZIm'] >= 0)]

        rows_to_keep = []

        # Always include last two points
        rows_to_keep.extend([len(df) - 1, len(df) - 2])

        # Loop from third-to-last down to index 2
        for i in range(len(df) - 3, 1, -1):
            second_previous_zre = df.iloc[i + 2]['ZRe']
            previous_zre = df.iloc[i + 1]['ZRe']
            current_zre = df.iloc[i]['ZRe']
            next_zre = df.iloc[i - 1]['ZRe']
            second_next_zre = df.iloc[i - 2]['ZRe']

            if previous_zre + 1 > current_zre > next_zre - 1:
                rows_to_keep.append(i)
            elif second_previous_zre + 1 > previous_zre or next_zre > second_next_zre - 1:
                rows_to_keep.append(i)

        # Always include first two points
        rows_to_keep.extend([0, 1])

        # Sort and deduplicate just in case
        rows_to_keep = sorted(set(rows_to_keep))

        # Extract cleaned x and y
        x = [df.iloc[i]["ZRe"] for i in rows_to_keep]
        y = [df.iloc[i]["ZIm"] for i in rows_to_keep]
        
        df = pd.DataFrame({'ZRe': x, 'ZIm': y})
        pointer = len(df) - 3
        max_lookahead = 3

        while pointer > 0:
            current_value = df.iloc[pointer]['ZRe']
            for offset in range(1, max_lookahead + 1):
                if pointer - offset < 0:
                    break
                next_value = df.iloc[pointer - offset]['ZRe']
                if (next_value -1) < current_value:
                    pointer -= offset
                    break
            else:
                break

        if pointer != 0:
            to_drop = list(range(0, pointer))
            df = df.drop(to_drop)

        x = df['ZRe'].tolist()
        y = df['ZIm'].tolist()
        
        for i in range(3):
            x_r = []
            y_r = []
            x_k = []
            y_k = []
            x_k.append(x[0])
            y_k.append(y[0])
            for idx in range(1, len(x) - 1):
                x_prev = x[idx - 1]
                y_prev = y[idx - 1]

                x_next = x[idx + 1]
                y_next = y[idx + 1]
                x_avg = (x_prev + x_next) / 2
                y_avg = (y_prev + y_next) / 2
                x_r.append(x)
                y_r.append(y)
                x_k.append(x_avg)
                y_k.append(y_avg)

            x_k.append(x[-1])
            y_k.append(y[-1])

            x = x_k
            y = y_k
        end_dict = {"ZRe": x, "ZIm": y}
        df = pd.DataFrame(end_dict)
        if len(df) < 4:
            return "Not enough data"

        x = df['ZRe'].iloc[:]
        y = df['ZIm'].iloc[:]

        good_impedance = False
        min_y = np.min(y)
        index = (np.abs(y - min_y)).argmin()
        min_x = x.iloc[index]
        if index > 1 and y.iloc[index - 1] > min_y and y.iloc[index - 2] > min_y:
            good_impedance = True

        if not good_impedance:
            coefficients = np.polyfit(x, y, 2)
            poly_2 = np.poly1d(coefficients)
            if coefficients[0] < 0:
                xn = np.linspace(15, np.max(x), 10)
                yn = np.poly1d(coefficients)(xn)
                roots = np.roots(poly_2)
                is_complex = np.iscomplex(roots)
                good_roots = roots[np.where(np.invert(is_complex))]
                if len(good_roots) > 0:
                    x_imag = np.min(good_roots)
                    x_real = np.real(x_imag)
                    return x_real
            else:
                return(min_x)
        else:

            min_index = df['ZIm'].idxmin()
            y_min = df['ZIm'].iloc[min_index]
            x_at_min_y = df['ZRe'].iloc[min_index]

            # plt.plot(df["ZRe"], df["ZIm"], marker="o")

            a = True
            b = False
            if min_index != 0:
                if df['ZIm'].iloc[-1] == y_min:
                    x_min = df['ZRe'].iloc[min_index]
                    a = False

                if df['ZIm'].iloc[-1] != y_min and df['ZIm'].iloc[-1] - y_min <= 0.5:
                    x_min = (df['ZRe'].iloc[min_index] + df['ZRe'].iloc[-1]) / 2
                    a = False

                if a == True:
                    x = df['ZRe'].iloc[:]
                    y = df['ZIm'].iloc[:]
                    poly_coeffs = np.polyfit(x, y, 4)
                    poly_4 = np.poly1d(poly_coeffs)
                    x_plot_p4 = np.linspace(df['ZRe'].iloc[0], df['ZRe'].iloc[-1], 25)
                    y_plot_p4 = poly_4(x_plot_p4)
                    min_index_p4 = np.argmin(y_plot_p4)
                    x_min_p4 = x_plot_p4[min_index_p4]

                    # Select the points: one before, the minimum, and one after
                    x_values = df['ZRe'].iloc[min_index - 1: min_index + 2]
                    y_values = df['ZIm'].iloc[min_index - 1: min_index + 2]

                    # Fit a second-degree polynomial (quadratic) through these three points
                    coefficients = np.polyfit(x_values, y_values, 2)
                    poly_2 = np.poly1d(coefficients)
                    # Get the coefficients of the polynomial
                    a, b, c = coefficients  # where coefficients = [a, b, c] from np.polyfit

                    # Calculate the x-coordinate of the minimum
                    x_min_p2 = -b / (2 * a)
                    x_plot_p2 = np.linspace(df['ZRe'].iloc[min_index - 1], df['ZRe'].iloc[min_index + 1], 20)
                    y_plot_p2 = poly_2(x_plot_p2)
                    b = True
                    a = True

            else:
                # Assuming `df` is your DataFrame and has columns 'ZRe' and 'ZIm'
                x = df['ZRe'].iloc[:]
                y = df['ZIm'].iloc[:]

                # Step 1: Fit a 4th order polynomial to the data
                poly_coeffs = np.polyfit(x, y, 4)
                poly_4 = np.poly1d(poly_coeffs)

                # Step 2: Compute the first derivative of the polynomial
                poly_4_deriv = poly_4.deriv()

                # Step 3: Find the roots of the derivative to locate points where the slope is zero
                critical_points = poly_4_deriv.roots
                real_critical_points = [point for point in critical_points if np.isreal(point)]
                real_critical_points = np.real(real_critical_points)  # Extract real parts

                # Filter critical points to only include those within the observed range of x values
                x_min_val, x_max_val = x.min(), x.max()
                real_critical_points = [point for point in real_critical_points if x_min_val <= point <= x_max_val]

                # Step 4: Check for local minima and maxima by evaluating the derivative around each critical point
                local_minima = []
                local_maxima = []

                for point in real_critical_points:
                    # Evaluate the derivative slightly to the left and right of the point
                    left_of_point = poly_4_deriv(point - 1e-4)
                    right_of_point = poly_4_deriv(point + 1e-4)

                    # Check if it is a local minimum (derivative changes from negative to positive)
                    if left_of_point < 0 and right_of_point > 0:
                        local_minima.append(point)

                    # Check if it is a local maximum (derivative changes from positive to negative)
                    if left_of_point > 0 and right_of_point < 0:
                        local_maxima.append(point)

                # Step 5: Process based on whether a local minimum was found
                if local_minima and local_maxima:

                    # Set x_min_p4 to the first local minimum that meets the criteria
                    x_min_p4 = local_minima[0]

                    lower_bound = local_minima[0] - 2
                    upper_bound = local_minima[0] + 6
                    df_in_range = df[(df['ZRe'] >= lower_bound) & (df['ZRe'] <= upper_bound)]
                    if not df_in_range.empty:
                        min_index = df_in_range['ZIm'].idxmin()
                        y_min = df_in_range['ZIm'].loc[min_index]
                        x_at_min_y = df_in_range['ZRe'].loc[min_index]

                    # Step 7: Fit a second-degree polynomial around x_at_min_y and calculate x_min_p2
                    # Ensure min_index has surrounding points to perform the fit
                    if min_index > 0 and min_index < len(df) - 1:
                        x_values = df['ZRe'].iloc[min_index - 1: min_index + 2]
                        y_values = df['ZIm'].iloc[min_index - 1: min_index + 2]

                        # Fit a second-degree polynomial (quadratic) through these three points
                        coefficients = np.polyfit(x_values, y_values, 2)
                        poly_2 = np.poly1d(coefficients)

                        # Calculate the x-coordinate of the minimum of the quadratic polynomial
                        a, b, c = coefficients  # where coefficients = [a, b, c] from np.polyfit
                        x_min_p2 = -b / (2 * a)

                        # Optional: generate values for plotting the second-degree polynomial
                        x_plot_p2 = np.linspace(df['ZRe'].iloc[min_index - 1], df['ZRe'].iloc[min_index + 1], 20)
                        y_plot_p2 = poly_2(x_plot_p2)
                        a = True
                        b = True

                    else:
                        logging.error(
                            "Not enough data points to fit a second-degree polynomial around the minimum ZIm.")


                else:
                    # Assuming 'df' is your DataFrame and has columns 'ZRe' and 'ZIm'
                    x = df['ZRe'].iloc[:]
                    y = df['ZIm'].iloc[:]

                    # Step 1: Fit a 3rd-order polynomial to the data
                    poly_coeffs = np.polyfit(x, y, 3)
                    poly_3 = np.poly1d(poly_coeffs)

                    # Step 2: Compute the first derivative of the polynomial
                    poly_3_deriv = poly_3.deriv()

                    # Step 3: Compute the second derivative of the polynomial (for finding critical points of the first derivative)
                    poly_3_deriv2 = poly_3_deriv.deriv()

                    # Step 4: Find the roots of the second derivative to locate critical points of the first derivative
                    critical_points = poly_3_deriv2.roots
                    real_critical_points = [point for point in critical_points if np.isreal(point)]
                    real_critical_points = np.real(real_critical_points)  # Extract real parts

                    # Step 5: Filter critical points within the observed range of x values
                    x_min_val, x_max_val = x.min(), x.max()
                    real_critical_points = [point for point in real_critical_points if x_min_val <= point <= x_max_val]

                    # Step 6: Identify the minimum of the first derivative by evaluating the third derivative at each critical point
                    poly_3_deriv3 = poly_3_deriv2.deriv()  # Third derivative for checking minima
                    x_min = None
                    min_deriv_value = float('inf')  # Start with a large value

                    for point in real_critical_points:
                        # Check if this point is a minimum of the first derivative
                        if poly_3_deriv3(point) > 0:  # Positive third derivative implies a local minimum
                            deriv_value = poly_3_deriv(point)  # Value of the first derivative at this point
                            if deriv_value < min_deriv_value:  # Find the smallest value of the first derivative
                                min_deriv_value = deriv_value
                                x_min = point

                    # Debug: Print the found minimum point of the first derivative
                    if x_min is not None:
                        logging.info(
                            f"Minimum of the first derivative found at x = {x_min}, with derivative value = {min_deriv_value}")
                    else:
                        logging.info("No minimum of the first derivative found within the x range.")

                    b = False
                    a = False
            if a == False:
                return x_min
            if b == True:

                avg = (x_at_min_y + x_min_p4 + x_min_p2) / 3
                data_R = [x_at_min_y, x_min_p4, x_min_p2]
                if abs(statistics.stdev(data_R) / avg) >= 0.08:
                    dev_p2 = abs(x_min_p2 - avg)
                    dev_p4 = abs(x_min_p4 - avg)
                    dev_min = abs(x_at_min_y - avg)
                    dev_avg = [dev_min, dev_p4, dev_p2]

                    max_dev_index = dev_avg.index(max(dev_avg))

                    # Remove the item in data_R at this index
                    data_R = [v for i, v in enumerate(data_R) if i != max_dev_index]
                    avg = sum(data_R) / len(data_R)
                return avg

    def extract_data_imp_second_semi(file_path: str, ir_result):

        # Data extraction for ZRe and ZIm
        zre_values, zim_values = [], []

        obj = DataExtractor.get_json_from_psfile(file_path)
        measurement_data = obj.get('Measurements', [])
        data_set = measurement_data[0].get('DataSet', {}) if measurement_data else {}
        values = data_set.get('Values', []) if data_set else []
        for value in values:
            # Both values are belived to be in ohms always
            if value.get('Description') == 'ZRe':
                zre_vals = value.get('DataValues', [])
                for pair in zre_vals:
                    zre_values.append(pair['V'])
            if value.get('Description') == 'ZIm':
                zim_vals = value.get('DataValues', [])
                for pair in zim_vals:
                    zim_values.append(pair['V'])
        # Create a DataFrame with columns ZRe and ZIm
        df = pd.DataFrame({'ZRe': zre_values, 'ZIm': zim_values})

        # Check if both ZRe and ZIm lists have values and are of the same length
        if zre_values and zim_values and len(zre_values) == len(zim_values):
            # Create a DataFrame with columns ZRe and ZIm
            df = pd.DataFrame({'ZRe': zre_values, 'ZIm': zim_values})

            # Remove rows where either ZRe or ZIm contains a negative value
            df = df[(df['ZRe'] >= 0) & (df['ZIm'] >= 0)]


        rows_to_keep = []

        # Always include last two points
        if len(df) >= 2:
            rows_to_keep.extend([len(df) - 1, len(df) - 2])

        # Loop from third-to-last to 2nd
        for i in range(len(df) - 3, 1, -1):
            second_previous_zre = df.iloc[i + 2]['ZRe']
            previous_zre = df.iloc[i + 1]['ZRe']
            current_zre = df.iloc[i]['ZRe']
            next_zre = df.iloc[i - 1]['ZRe']
            second_next_zre = df.iloc[i - 2]['ZRe']

            if previous_zre + 1 > current_zre > next_zre - 1:
                rows_to_keep.append(i)
            elif second_previous_zre + 1 > previous_zre or next_zre > second_next_zre - 1:
                rows_to_keep.append(i)

        # Always include first two points
        rows_to_keep.extend([0, 1])
        rows_to_keep = sorted(set(rows_to_keep))

        # Apply ZRe filter
        x = [df.iloc[i]["ZRe"] for i in rows_to_keep]
        y = [df.iloc[i]["ZIm"] for i in rows_to_keep]
        df = pd.DataFrame({'ZRe': x, 'ZIm': y})

        pointer = len(df) - 3
        max_lookahead = 3

        while pointer > 0:
            current_value = df.iloc[pointer]['ZRe']
            for offset in range(1, max_lookahead + 1):
                if pointer - offset < 0:
                    break
                next_value = df.iloc[pointer - offset]['ZRe']
                if (next_value - 1) < current_value:
                    pointer -= offset
                    break
            else:
                break

        if pointer != 0:
            to_drop = list(range(0, pointer))
            df = df.drop(to_drop).reset_index(drop=True)

        x = df['ZRe'].tolist()
        y = df['ZIm'].tolist()

        for i in range(3):
            x_k = [x[0]]
            y_k = [y[0]]
            for idx in range(1, len(x) - 1):
                x_avg = (x[idx - 1] + x[idx + 1]) / 2
                y_avg = (y[idx - 1] + y[idx + 1]) / 2
                x_k.append(x_avg)
                y_k.append(y_avg)
            x_k.append(x[-1])
            y_k.append(y[-1])
            x = x_k
            y = y_k

        df = pd.DataFrame({'ZRe': x, 'ZIm': y})
        
        if len(df) >= 6:
            x = range(2, 6)
            y = df['ZIm'].iloc[2:6]
            poly_coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(poly_coeffs)

            # Check if the first or first two points deviate from the trendline by more than ±1
            outliers = []
            for i in range(2):  # Only check the first two points
                if abs(df.iloc[i]['ZIm'] - poly(i)) > 1:
                    outliers.append(i)

            # Remove identified outliers
            if outliers:
                df = df.drop(outliers).reset_index(drop=True)

            # Additional filtering condition: Keep rows where ZRe > result
            cut_index = None
            for i in reversed(range(len(df))):
                if df.iloc[i]['ZRe'] <= ir_result:
                    cut_index = i + 1
                    break

            if cut_index is not None and cut_index < len(df):
                df = df.iloc[cut_index:].reset_index(drop=True)
            else:
                df = df.reset_index(drop=True)

            if len(df) < 4:
                logging.error("Not enough data points to fit a second-degree polynomial.")
                return

            # Fit a second-degree polynomial
            x = df['ZRe']
            y = df['ZIm']
            poly_coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(poly_coeffs)

            # Find the maximum of the polynomial
            a, b, c = poly_coeffs  # Coefficients of the polynomial
            if a < 0:  # Check if the parabola opens downward
                x_max = -b / (2 * a)  # x-coordinate of the vertex
                y_max = poly(x_max)  # y-coordinate (maximum value)
                return x_max  # Return the maximum point (x, y)
            else:
                logging.error("Parabola does not open downward.")
                return

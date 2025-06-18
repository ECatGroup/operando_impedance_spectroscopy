from state import STATE


class UpdateParameterFile:
    """
    Handles reading, updating, and writing parameter values in a specified file.

    This class is designed to manage numerical and string parameters by reading an existing
    file, updating specific parameter values, and writing the updated contents back to the
    file. Parameters can be formatted in scientific notation if they are numeric. It includes
    mechanisms to ensure parameters are found and updated accordingly while preserving other
    file contents.

    :ivar parameter_file_path: The file path to the parameter file that will be updated.
    :type parameter_file_path: str
    :ivar parameters: Dictionary of parameter names and their corresponding values to be updated.
    :type parameters: dict
    :ivar found_parameters: Dictionary indicating whether each parameter was found in the file.
    :type found_parameters: dict
    :ivar lines: List of all lines read from the file.
    :type lines: list
    """
    def __init__(self, parameter_file_path: str):

        self.parameter_file_path = parameter_file_path
        self.parameters = {}
        self.found_parameters = {}
        self.lines = []

    def set_parameter(self, name, value):
        """
        Sets a parameter in the `parameters` dictionary of the object. If the
        given value is a number (either integer or floating-point), it formats
        it using scientific notation with three significant digits. Otherwise,
        the value is stored as is.

        :param name: The name of the parameter to set.
        :type name: str
        :param value: The value to associate with the parameter name. If numeric
            (int or float), the value is formatted in scientific notation
            (3 significant digits); otherwise, it remains unchanged.
        :return: The current instance of the class allowing method chaining.
        :rtype: object
        """
        if value is not None:
            if isinstance(value, (int, float)):
                self.parameters[name] = f"{value:.3E}"
            else:
                self.parameters[name] = value
        return self

    def _read_file(self):
        """
        Reads the contents of a file specified in the `parameter_file_path` attribute
        and assigns the lines to the `lines` attribute. Handles potential exceptions
        that may occur during the file reading process and logs the error if an
        exception is raised.

        The file reading process uses UTF-16 encoding and assumes the file exists
        and is accessible. Returns a boolean indicating the success or failure
        of the file reading operation.

        :raises OSError: If there are any issues opening or reading the file.
        :raises ValueError: If the file's encoding does not match the specified encoding.
        :return: A boolean indicating success (True) or failure (False)
        :rtype: bool
        """
        try:
            with open(self.parameter_file_path, 'r', encoding='utf-16') as file:
                self.lines = file.readlines()
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False

    def _update_lines(self):
        """
        Updates the lines in self.lines based on the current parameters in self.parameters.

        The `_update_lines` method processes the lines in the `self.lines` list and updates them
        to reflect the values specified in the `self.parameters` dictionary. It additionally
        tracks which parameters have been processed and appropriately modifies the corresponding
        entries in self.found_parameters. Special handling is implemented for the "E_VALUE"
        parameter to account for its variations ("E=" and "E_BEGIN=").

        :raises KeyError: If an undefined parameter key is encountered in `self.parameters`.
        """
        self.found_parameters = {key: False for key in self.parameters}

        for i in range(len(self.lines)):
            line = self.lines[i].strip()

            # Handle E/E_BEGIN as a special case
            if "E_VALUE" in self.parameters:
                if line.startswith("E="):
                    self.lines[i] = f"E={self.parameters['E_VALUE']}\n"
                    self.found_parameters["E_VALUE"] = True
                elif line.startswith("E_BEGIN="):
                    self.lines[i] = f"E_BEGIN={self.parameters['E_VALUE']}\n"
                    self.found_parameters["E_VALUE"] = True

            # Handle all other parameters
            for param, formatted_value in self.parameters.items():
                if param == "E_VALUE":  # Already handled above
                    continue
                param_prefix = {
                    "IR_DROP": "IR_DROP_COMP_RES",
                    "T_RUN": "T_RUN",
                    "CURRENT": "CURRENT",
                    "E_STBY": "E_STBY"
                }.get(param, param)

                if line.startswith(f"{param_prefix}="):
                    self.lines[i] = f"{param_prefix}={formatted_value}\n"
                    self.found_parameters[param] = True

    def _write_file(self):
        """
        Writes parameter values to a file and prints updates for modified parameters.

        This method writes the content of `self.lines` to the file specified by
        `self.parameter_file_path` using UTF-16 encoding. It maps specific parameter
        names to more meaningful names, prints an update message for each mapped parameter
        that has been modified in `self.parameters`, and logs errors in case of failure.

        :return: Returns a boolean indicating success (`True`) or failure (`False`) of the file-writing operation.
        :rtype: bool
        """
        try:
            with open(self.parameter_file_path, 'w', encoding='utf-16') as file:
                file.writelines(self.lines)

            # Print update messages
            for param, found in self.found_parameters.items():
                if found:
                    param_name = {
                        "IR_DROP": "IR_DROP_COMP_RES",
                        "T_RUN": "T_RUN",
                        "E_VALUE": "E or E_BEGIN",
                        "CURRENT": "CURRENT",
                        "E_STBY": "E_STBY"
                    }.get(param, param)
                    if STATE.debug:
                        print(f"Updated {param_name} to {self.parameters[param]}")
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False

    def update(self):
        """
        Updates parameters by reading a file, processing the parameters, and rewriting the file if any
        parameters were successfully updated.

        The method sequentially checks if parameters exist, reads the file, updates lines with modifications
        based on the parameters, and writes back to the file if updates were made. If no parameters are found
        or updated, an appropriate message is printed, and the method returns `False`.

        :returns:
            - ``True`` if the file is successfully updated with the new parameters.
            - ``False`` if no parameters were found or updated, the file could not be read, or any other
              issue occurs during the update process.
        :rtype: bool
        """
        if not self.parameters:
            print("No parameters to update")
            return False

        if not self._read_file():
            return False

        self._update_lines()

        # Check if any parameters were found and updated
        if any(self.found_parameters.values()):
            return self._write_file()
        else:
            print("No matching parameters found in the file")
            return False

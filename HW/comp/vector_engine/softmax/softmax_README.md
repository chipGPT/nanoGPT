# Verilog Softmax Module

## Overview

This section implements a **Softmax Function** in Verilog, typically used in machine learning for normalizing an input vector into probabilities. The softmax module takes floating-point inputs and outputs normalized probabilities. A testbench is also provided to simulate and validate the module.

## Files

- **softmax.v**: This file contains the Verilog implementation of the softmax function. It computes the softmax for a given set of inputs using a series of floating-point arithmetic operations.

- **softmax_tb.v**: This is the testbench file for simulating and verifying the functionality of the `softmax.v` module. The testbench generates various input vectors and checks if the outputs are correct by comparing them with expected results.

## Simulation

To run the simulation and verify the functionality of the softmax module:

1. Make sure you have a Verilog simulator installed (e.g., ModelSim, Xilinx Vivado, Icarus Verilog).
2. Compile the `softmax.v` and `softmax_tb.v` files in your simulation tool.
3. Run the testbench (`softmax_tb.v`) to observe the results.

Microwave Filter Coupling Matrix Tuning Tool (Pythonista / iPad)

An interactive coupling-matrix design and tuning tool for microwave bandpass filters, built for Pythonista on iPad. It synthesizes Chebyshev or Butterworth prototypes, plus filters with finite transmission zeros, using a polynomial-based folded-form synthesis. Each design starts from an immutable "golden" matrix. You tune a working copy through an editor with undo, reset, and randomize.

Features

	•	Live S11/S21/S22 simulation including finite-Qu loss
	•	Insertion loss, phase, and group-delay plots, with a Smith chart option
	•	Three markers with delta readouts, and a golden-response overlay
	•	Import and export Touchstone .s2p files, with measurement overlay and de-embedding
	•	Diagnostics (reciprocity, energy conservation, TZ synthesis checks), approximate poles, and ±5% sensitivity analysis
	•	Extraction of Qe and coupling values against the design

Stack: NumPy, Matplotlib (Agg backend), and Pythonista's ui and dialogs modules.

Usage: Open the script in Pythonista and run it in landscape. Files are saved to ~/Documents/S2P.
